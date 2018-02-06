#include <float.h>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <vector>
#include <queue>
#include <list>
#include <chrono>
#include <random>

#include "reductions.h"
#include "vw.h"

using namespace std;
using namespace LEARNER;

#define DEBUG false
#define D_COUT if(DEBUG) cout

struct node{
    uint32_t n;
    float p;

    bool operator < (const node& r) const { return p < r.p; }
};

struct hsm {
    vw* all;

    uint32_t k; // number of labels
    uint32_t t; // number of tree nodes
    uint32_t ti; // number of internal nodes
    uint32_t kary;

    uint32_t *labels_nodes_map;
    uint32_t *nodes_labels_map;
    bool positive_labels;   // print positive labels
    bool remap_labels;
    float* nodes_t;
    uint32_t p_at_k;

    float precision;
    float predicted_number;
    vector<float> precision_at_k;
    size_t prediction_count;

    //bool inference_threshold;
    //bool inference_top_k;

    long n_visited_nodes;

    float ec_loss;
    float loss_sum;
    uint32_t ec_count;
    chrono::time_point<chrono::steady_clock> learn_predict_start_time_point;
    default_random_engine rng;
};


// helpers
//----------------------------------------------------------------------------------------------------------------------

inline float sigmoid(float in) { return 1.0f / (1.0f + exp(-in)); }


// save/load
//----------------------------------------------------------------------------------------------------------------------

void save_load_nodes(hsm& p, io_buf& model_file, bool read, bool text){

    D_COUT << "SAVE/LOAD TREE\n";

    if (model_file.files.size() > 0) {
        bool resume = p.all->save_resume;
        stringstream msg;
        msg << ":" << resume << "\n";
        bin_text_read_write_fixed(model_file, (char*) &resume, sizeof(resume), "", read, msg, text);

        if(resume){
            for(size_t i = 0; i < p.t; ++i)
                bin_text_read_write_fixed(model_file, (char *) &p.nodes_t[i], sizeof(p.nodes_t[0]), "", read, msg, text);
        }
    }
}


// learn
//----------------------------------------------------------------------------------------------------------------------

void learn_node(hsm& p, uint32_t n, base_learner& base, example& ec){
    D_COUT << "LEARN NODE: " << n << " LABEL: " << ec.l.simple.label << " WEIGHT: " << ec.weight << " NODE_T: " << p.nodes_t[n] << endl;

    ec.loss = 0;
    p.all->sd->t = p.nodes_t[n];
    p.nodes_t[n] += ec.weight;

    base.learn(ec, n);
    ++p.n_visited_nodes;
    p.ec_loss += ec.loss;
}

//K-ARY
void learn(hsm& p, base_learner& base, example& ec){

    D_COUT << "LEARN EXAMPLE: " << p.all->sd->example_number << " PASS: " << p.all->passes_complete << endl;

    COST_SENSITIVE::label ec_labels = ec.l.cs;
    double t = p.all->sd->t;
    double weighted_holdout_examples = p.all->sd->weighted_holdout_examples;
    p.all->sd->weighted_holdout_examples = 0;

    if (ec_labels.costs.size() > 0) {

        uniform_int_distribution<size_t> uniform(0, ec_labels.costs.size()-1);

        uint32_t ec_label = 0;
        uint32_t ec_l_idx = uniform(p.rng);

        uint32_t i = 0;
        for (auto& cl : ec_labels.costs) {
            ec_label = cl.class_index;
            if(i++ == ec_l_idx) break;
        }

        if (ec_label > p.k)
            cerr << "Label " << ec_label << " is not in {1," << p.k << "} This won't work right." << endl;
        uint32_t tn = ec_label + p.ti - 1;

        ec.l.simple = {1.f, 0.f, 0.f};
        while (tn > 0) {
            uint32_t new_tn = floor(static_cast<float>(tn - 1) / p.kary);
            for(int child_index = 1; child_index <= p.kary; child_index++){
                int child_tn = p.kary*new_tn + child_index;
                if(child_tn < p.t){
                    ec.l.simple.label = (child_tn == tn ? 1.0: -1.0);
                    learn_node(p, child_tn, base, ec);
                }
            }
            tn = new_tn;
        }
    }
    else
        cerr << "No label, this won't work right." << endl;

    p.ec_loss = 0;

    p.loss_sum += p.ec_loss;
    ec.loss = p.ec_loss;
    p.ec_count += 1;

    ec.l.cs = ec_labels;
    p.all->sd->t = t;
    p.all->sd->weighted_holdout_examples = weighted_holdout_examples;
    ec.pred.multiclass = 0;


    D_COUT << "----------------------------------------------------------------------------------------------------\n";
}


////BINARY
//void learn(hsm& p, base_learner& base, example& ec){
//
//    D_COUT << "LEARN EXAMPLE: " << p.all->sd->example_number << " PASS: " << p.all->passes_complete << endl;
//
//    COST_SENSITIVE::label ec_labels = ec.l.cs;
//    double t = p.all->sd->t;
//    double weighted_holdout_examples = p.all->sd->weighted_holdout_examples;
//    p.all->sd->weighted_holdout_examples = 0;
//
//    if (ec_labels.costs.size() > 0) {
//
//        uniform_int_distribution<size_t> uniform(0, ec_labels.costs.size()-1);
//
//        uint32_t ec_label = 0;
//        uint32_t ec_l_idx = uniform(p.rng);
//
//        uint32_t i = 0;
//        for (auto& cl : ec_labels.costs) {
//            ec_label = cl.class_index;
//            if(i++ == ec_l_idx) break;
//        }
//
//        if (ec_label > p.k)
//            cerr << "Label " << ec_label << " is not in {1," << p.k << "} This won't work right." << endl;
//        uint32_t tn = ec_label + p.ti - 1;
//
//        ec.l.simple = {1.f, 0.f, 0.f};
//        while (tn > 0) {
//            uint32_t new_tn = floor(static_cast<float>(tn - 1) / p.kary);
//            ec.l.simple.label = (tn % p.kary != 0 ? 1.0: -1.0);
//            learn_node(p, new_tn, base, ec);
//            tn = new_tn;
//        }
//    }
//    else
//        cerr << "No label, this won't work right." << endl;
//
//    p.loss_sum += p.ec_loss;
//    p.ec_count++;
//    ec.l.cs = ec_labels;
//    ec.loss = p.ec_loss;
//
//    p.all->sd->t = t;
//    p.all->sd->weighted_holdout_examples = weighted_holdout_examples;
//    ec.pred.multiclass = 0;
//
//
//    D_COUT << "----------------------------------------------------------------------------------------------------\n";
//}

// predict
//----------------------------------------------------------------------------------------------------------------------

inline float predict_node(hsm& p, uint32_t n, base_learner& base, example& ec){
    D_COUT << "PREDICT NODE: " << n << endl;

    ec.l.simple = {FLT_MAX, 0.f, 0.f};
    base.predict(ec, n);
    ++p.n_visited_nodes;

    return sigmoid(ec.partial_prediction);
}



//K-ARY
void predict(hsm& p, base_learner& base, example& ec){

    D_COUT << "PREDICT EXAMPLE: " << p.all->sd->example_number << endl;

    ++p.prediction_count;

    COST_SENSITIVE::label ec_labels = ec.l.cs;

    vector<uint32_t> best_labels, found_leaves;
    priority_queue<node> node_queue;
    node_queue.push({0, 1.0f});

    while (!node_queue.empty()) {
        node node = node_queue.top(); // current node
        node_queue.pop();

        if (node.n >= p.ti) {
            uint32_t l = p.nodes_labels_map[node.n - p.ti + 1];
            best_labels.push_back(l);
            if (best_labels.size() >= p.p_at_k) break;
        } else {
            vector<float> node_probabs;
            float proba_sum = 0.0;
            for(int child_index = 1; child_index <= p.kary; child_index++){
                int child_tn = p.kary*node.n + child_index;

                if(child_tn < p.t){
                    float proba =  predict_node(p, child_tn, base, ec);
                    node_probabs.push_back(proba);
                    proba_sum += proba;
                }
            }
            for(int i = 0; i<node_probabs.size(); i++){
                node_probabs[i] /= proba_sum;
            }
            for(int child_index = 1; child_index <= p.kary; child_index++){
                int child_tn = p.kary*node.n + child_index;
                if(child_tn < p.t) {
                    node_queue.push({child_tn, node.p*node_probabs[child_index - 1]});
                }
            }

        }
    }

    vector<uint32_t> true_labels;
    for (auto &cl : ec_labels.costs) true_labels.push_back(cl.class_index);

    if (p.p_at_k > 0 && true_labels.size() > 0) {
        for (size_t i = 0; i < p.p_at_k; ++i) {
            if (find(true_labels.begin(), true_labels.end(), best_labels[i]) != true_labels.end())
                p.precision_at_k[i] += 1.0f;
        }
    }

    ec.l.cs = ec_labels;

    D_COUT << "----------------------------------------------------------------------------------------------------\n";
}


////BINARY
//void predict(hsm& p, base_learner& base, example& ec){
//
//    D_COUT << "PREDICT EXAMPLE: " << p.all->sd->example_number << endl;
//
//    ++p.prediction_count;
//
//    COST_SENSITIVE::label ec_labels = ec.l.cs;
//
//    vector<uint32_t> best_labels, found_leaves;
//    priority_queue<node> node_queue;
//    node_queue.push({0, 1.0f});
//
//    while (!node_queue.empty()) {
//        node node = node_queue.top(); // current node
//        node_queue.pop();
//
//        if (node.n >= p.ti) {
//            uint32_t l = p.nodes_labels_map[node.n - p.ti + 1];
//            best_labels.push_back(l);
////            cout<<l<<" : "<< node.p<<"; ";
//            if (best_labels.size() >= p.p_at_k) break;
//        } else {
//            float p_left =  predict_node(p, node.n, base, ec);
//
//            uint32_t n_child = p.kary * node.n + 1;
//            node_queue.push({n_child, node.p*p_left});
//
//            n_child = p.kary * node.n + 2;
//            node_queue.push({n_child, node.p*(1.0 - p_left)});
//
//        }
//    }
//
////    cout<<endl;
//    vector<uint32_t> true_labels;
//    for (auto &cl : ec_labels.costs) true_labels.push_back(cl.class_index);
//
//    if (p.p_at_k > 0 && true_labels.size() > 0) {
//        for (size_t i = 0; i < p.p_at_k; ++i) {
//            if (find(true_labels.begin(), true_labels.end(), best_labels[i]) != true_labels.end())
//                p.precision_at_k[i] += 1.0f;
//        }
//    }
//
//    ec.l.cs = ec_labels;
//
//    D_COUT << "----------------------------------------------------------------------------------------------------\n";
//}



// other
//----------------------------------------------------------------------------------------------------------------------

void finish_example(vw& all, hsm& p, example& ec){

    D_COUT << "FINISH EXAMPLE\n";

    /* TODO: find a better way to do it
    if(p.positive_labels) {
        char temp_str[10];
        ostringstream output_stream;
        for (size_t i = 0; i < positive_labels.size(); ++i) {
            if (i > 0) output_stream << ' ';
            output_stream << positive_labels[i].l;

            sprintf(temp_str, "%f", positive_labels[i].p);
            output_stream << ':' << temp_str;
        }
        for (int sink : all.final_prediction_sink)
            all.print_text(sink, output_stream.str(), ec.tag);

        
    }
    */

    all.sd->update(ec.test_only, 0.0f, ec.weight, ec.num_features);
    VW::finish_example(all, &ec);
}

void pass_end(hsm& p){
    cout << "end of pass " << p.all->passes_complete << ", avg. loss = " << p.loss_sum / p.ec_count << "\n";
}

template<bool use_threshold>
void finish(hsm& p){

    auto end_time_point = chrono::steady_clock::now();
    auto execution_time = end_time_point - p.learn_predict_start_time_point;
    cout << "learn_predict_time = " << static_cast<double>(chrono::duration_cast<chrono::microseconds>(execution_time).count()) / 1000000 << "s\n";

    // threshold prediction
    if (use_threshold) {
        if (p.predicted_number > 0) {
            cout << "Precision = " << p.precision / p.predicted_number << "\n";
        } else {
            cout << "Precision unknown - nothing predicted" << endl;
        }
    }

    // top-k predictions
    else {
        float correct = 0;
        for (size_t i = 0; i < p.p_at_k; ++i) {
            correct += p.precision_at_k[i];
            cout << "P@" << i + 1 << " = " << correct / (p.prediction_count * (i + 1)) << "\n";
        }
    }

    cout << "visited nodes = " << p.n_visited_nodes << endl;
    free(p.nodes_t);
}


// setup
//----------------------------------------------------------------------------------------------------------------------

base_learner* hsm_setup(vw& all) //learner setup
{
    if (missing_option<size_t, true>(all, "hsm", "Use probabilistic label tree for multilabel with <k> labels"))
        return nullptr;
    new_options(all, "hsm options")
        ("kary_tree", po::value<uint32_t>(), "tree in which each node has no more than k children")
        ("p_at", po::value<uint32_t>(), "P@k (default 1)")
	    ("positive_labels", "print all positive labels")
        ("remap_labels", "remap labels");
	
			
    add_options(all);

    hsm& data = calloc_or_throw<hsm>();
    data.k = (uint32_t)all.vm["hsm"].as<size_t>();
    data.remap_labels = false;
    data.all = &all;

    data.precision = 0;
    data.predicted_number = 0;
    data.prediction_count = 0;
    data.p_at_k = 1;
    data.n_visited_nodes = 0;
    data.loss_sum = 0.0;
    data.ec_count = 0;

    data.rng.seed(all.random_seed);

    // hsm parse options
    //------------------------------------------------------------------------------------------------------------------

    learner<hsm> *l;

    // ignored
    // kary options
    if(all.vm.count("kary_tree")) {
        data.kary = all.vm["kary_tree"].as<uint32_t>();

        double a = pow(data.kary, floor(log(data.k) / log(data.kary)));
        double b = data.k - a;
        double c = ceil(b / (data.kary - 1.0));
        double d = (data.kary * a - 1.0)/(data.kary - 1.0);
        double e = data.k - (a - c);
        data.t = static_cast<uint32_t>(e + d);
    }
    else{
        data.kary = 2;
        data.t = 2 * data.k - 1;
    }


    data.kary = 2;
    data.t = 2 * data.k - 1;

    data.ti = data.t - data.k;
    *(all.file_options) << " --kary_tree " << data.kary;

    if( all.vm.count("p_at") )
        data.p_at_k = all.vm["p_at"].as<uint32_t>();

    if( all.vm.count("positive_labels"))
        data.positive_labels = true;

    if( all.vm.count("remap_labels"))
        data.remap_labels = true;

    // init multiclass learner
    // -----------------------------------------------------------------------------------------------------------------

    data.precision_at_k.resize(data.p_at_k);
    l = &init_multiclass_learner(&data, setup_base(all), learn, predict, all.p, data.ti);
    l->set_finish(finish<false>);

    data.nodes_t = calloc_or_throw<float>(data.t);
    for(size_t i = 0; i < data.t; ++i) data.nodes_t[i] = all.initial_t;

    // map labels
    data.labels_nodes_map = calloc_or_throw<uint32_t>(data.k + 1);
    data.nodes_labels_map = calloc_or_throw<uint32_t>(data.k + 1);
    for(uint32_t i = 0; i <= data.k; ++i){
        data.labels_nodes_map[i] = i;
        data.nodes_labels_map[i] = i;
    }

    if(data.remap_labels){
        default_random_engine rng;
        rng.seed(all.random_seed);
        for(uint32_t i = 0; i <= data.k; ++i) {
            uniform_int_distribution <uint32_t> dist(i, data.k);
            auto swap = dist(rng);

            auto temp = data.labels_nodes_map[i];
            data.labels_nodes_map[i] = data.labels_nodes_map[swap];
            data.labels_nodes_map[swap] = temp;
        }

        for(uint32_t i = 0; i <= data.k; ++i) {
            data.nodes_labels_map[data.labels_nodes_map[i]] = i;
        }

        *(all.file_options) << " --remap_labels ";
        *(all.file_options) << " --random_seed " << all.random_seed;
    }

    //for(uint32_t i = 0; i <= data.k; ++i){
    //    cout << data.labels_nodes_map[i] << " " << data.nodes_labels_map[i] << endl;
    //}

    // override parser
    //------------------------------------------------------------------------------------------------------------------

    all.p->lp = COST_SENSITIVE::cs_label;
    all.cost_sensitive = make_base(*l);

    all.holdout_set_off = true; // turn off stop based on holdout loss

    // log info & add some event handlers
    //------------------------------------------------------------------------------------------------------------------
    cout << "hsm\n" << "k = " << data.k << "\ntree size = " << data.t << "\nkary_tree = " << data.kary << endl;

    l->set_finish_example(finish_example);
    l->set_save_load(save_load_nodes);
    l->set_end_pass(pass_end);

    return all.cost_sensitive;
}
