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

struct plt {
    vw* all;

    uint32_t k; // number of labels
    uint32_t t; // number of tree nodes
    uint32_t ti; // number of internal nodes
    uint32_t kary;

    float inner_threshold;  // inner threshold
    bool positive_labels;   // print positive labels
    bool top_k_labels;   // print top-k labels
    bool greedy;
    float* nodes_t;
    uint32_t p_at_k;
    float precision;
    float predicted_number;
    vector<float> precision_at_k;
    size_t prediction_count;

    //bool inference_threshold;
    //bool inference_top_k;
    
    long n_visited_nodes;

    uint32_t ec_count;
    chrono::time_point<chrono::steady_clock> learn_predict_start_time_point;
};

// debug helpers - to delete
//----------------------------------------------------------------------------------------------------------------------

void plt_example_info(plt& p, base_learner& base, example& ec){
//    cout << "TAG: " << (ec.tag.size() ? std::string(ec.tag.begin()) : "-") << " FEATURES COUNT: " << ec.num_features << " LABELS COUNT: " << ec.l.cs.costs.size() << endl;
//
//    cout << "BW: " << base.weights << " BI: " << base.increment
//         << " WSS: " << p.all->weights.stride_shift() << " WM: " << p.all->weights.mask() << endl;
//
//    for (features &fs : ec) {
//        for (features::iterator_all &f : fs.values_indices_audit())
//            cout << "FEATURE: " << (f.index() & p.all->weights.mask()) << " VALUE: " << f.value() << endl;
//    }
//    for (auto &cl : ec.l.cs.costs) cout << "LABEL: " << cl.class_index << endl;
}

void plt_prediction_info(base_learner& base, example& ec){
//    cout << std::fixed << std::setprecision(6) << "PP: " << ec.partial_prediction << " UP: " << ec.updated_prediction
//         << " L: " << ec.loss << " S: " << ec.pred.scalar << endl;
}

void plt_print_all_weights(plt &p){
//    cout << endl << "W:";
//    for (uint64_t i = 0; i <= p.all->weights.mask() + 10; ++i) {
//        cout << " " << p.all->weights.first()[i];
//    }
//    cout << endl;
}


// helpers
//----------------------------------------------------------------------------------------------------------------------

inline float sigmoid(float in) { return 1.0f / (1.0f + exp(-in)); }


// save/load
//----------------------------------------------------------------------------------------------------------------------

void save_load_nodes(plt& p, io_buf& model_file, bool read, bool text){

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

void learn_node(plt& p, uint32_t n, base_learner& base, example& ec){
    D_COUT << "LEARN NODE: " << n << " LABEL: " << ec.l.simple.label << " WEIGHT: " << ec.weight << " NODE_T: " << p.nodes_t[n] << endl;

    p.all->sd->t = p.nodes_t[n];
    p.nodes_t[n] += ec.weight;

    base.learn(ec, n);
    ++p.n_visited_nodes;
}

void learn(plt& p, base_learner& base, example& ec){

    D_COUT << "LEARN EXAMPLE: " << p.all->sd->example_number << " PASS: " << p.all->passes_complete << endl;
    if(DEBUG) plt_example_info(p, base, ec);

    COST_SENSITIVE::label ec_labels = ec.l.cs;
    double t = p.all->sd->t;
    double weighted_holdout_examples = p.all->sd->weighted_holdout_examples;
    p.all->sd->weighted_holdout_examples = 0;

    unordered_set<uint32_t> n_positive; // positive nodes
    unordered_set<uint32_t> n_negative; // negative nodes

    if (ec_labels.costs.size() > 0) {
        for (auto& cl : ec_labels.costs) {
            if (cl.class_index > p.k)
                cerr << "Label " << cl.class_index << " is not in {1," << p.k << "} This won't work right." << endl;

            uint32_t tn = cl.class_index + p.ti - 1;
            n_positive.insert(tn);
            while (tn > 0) {
                tn = floor(static_cast<float>(tn - 1) / p.kary);
                n_positive.insert(tn);
            }
        }

        queue<uint32_t> n_queue; // nodes queue
        n_queue.push(0);

        while(!n_queue.empty()) {
            uint32_t n = n_queue.front(); // current node index
            n_queue.pop();

            if (n < p.ti) {
                for(uint32_t i = 1; i <= p.kary; ++i) {
                    uint32_t n_child = p.kary * n + i;

                    if(n_child < p.t) {
                        if (n_positive.find(n_child) != n_positive.end()) n_queue.push(n_child);
                        else n_negative.insert(n_child);
                    }
                }
            }
        }
    }
    else
        n_negative.insert(0);

    ec.l.simple = {1.f, 0.f, 0.f};
    for (auto &n : n_positive) learn_node(p, n, base, ec);

    ec.l.simple.label = -1.f;
    for (auto &n : n_negative) learn_node(p, n, base, ec);

    ec.l.cs = ec_labels;
    p.all->sd->t = t;
    p.all->sd->weighted_holdout_examples = weighted_holdout_examples;
    ec.pred.multiclass = 0;

    D_COUT << "----------------------------------------------------------------------------------------------------\n";
}


// predict
//----------------------------------------------------------------------------------------------------------------------

inline float predict_node(plt& p, uint32_t n, base_learner& base, example& ec){
    D_COUT << "PREDICT NODE: " << n << endl;

    ec.l.simple = {FLT_MAX, 0.f, 0.f};
    base.predict(ec, n);
    ++p.n_visited_nodes;

    return sigmoid(ec.partial_prediction);
}

template<bool use_threshold, bool greedy>
void predict(plt& p, base_learner& base, example& ec){

    D_COUT << "PREDICT EXAMPLE: " << p.all->sd->example_number << endl;
    if(DEBUG) plt_example_info(p, base, ec);

    ++p.prediction_count;

    COST_SENSITIVE::label ec_labels = ec.l.cs;

    // threshold prediction
    if (use_threshold) {
        vector<node> positive_labels;
        queue<node> node_queue;
        node_queue.push({0, 1.0f});

        while(!node_queue.empty()) {
            node node = node_queue.front(); // current node
            node_queue.pop();

            float cp = node.p * predict_node(p, node.n, base, ec);

            if(cp > p.inner_threshold){
                if (node.n < p.ti) {
                    for(uint32_t i = 1; i <= p.kary; ++i) {
                        uint32_t n_child = p.kary * node.n + i;
                        node_queue.push({n_child, cp});
                    }
                } else {
                    uint32_t l = node.n - p.ti + 1;
                    positive_labels.push_back({l, cp});
                }
            }
        }

        sort(positive_labels.rbegin(), positive_labels.rend());

        if (p.p_at_k > 0 && ec_labels.costs.size() > 0) {
            for (size_t i = 0; i < p.p_at_k && i < positive_labels.size(); ++i) {
                p.predicted_number += 1.0f;
                for (auto &cl : ec_labels.costs) {
                    if (positive_labels[i].n == cl.class_index) {
                        p.precision += 1.0f;
                        break;
                    }
                }
            }
        }
    }

    else if (greedy) {
        uint32_t current = 0;
        float current_p = predict_node(p, current, base, ec);

        while(current < p.ti){
            uint32_t best = p.kary * current + 1;
            float best_p = current_p * predict_node(p, best, base, ec);

            for(uint32_t i = 2; i <= p.kary; ++i) {
                uint32_t child = p.kary * current + i;
                float child_p = current_p * predict_node(p, child, base, ec);

                if(best_p < child_p){
                    best = child;
                    best_p = child_p;
                }
            }

            current = best;
            current_p = best_p;
        }

        vector<uint32_t> true_labels;
        for (auto &cl : ec_labels.costs) true_labels.push_back(cl.class_index);

        uint32_t label = current - p.ti + 1;
        if (find(true_labels.begin(), true_labels.end(), label) != true_labels.end())
            p.precision_at_k[0] += 1.0f;
    }

    // top-k predictions
    else {
        vector<uint32_t> best_labels, found_leaves;
        priority_queue<node> node_queue;
        node_queue.push({0, 1.0f});

        while (!node_queue.empty()) {
            node node = node_queue.top(); // current node
            node_queue.pop();

            if (find(found_leaves.begin(), found_leaves.end(), node.n) != found_leaves.end()) {
                uint32_t l = node.n - p.ti + 1;
                best_labels.push_back(l);
                if (best_labels.size() >= p.p_at_k) break;
            } else {
                float cp = node.p * predict_node(p, node.n, base, ec);
		
                if (node.n < p.ti) {
                    for(uint32_t i = 1; i <= p.kary; ++i) {
                        uint32_t n_child = p.kary * node.n + i;
                        node_queue.push({n_child, cp});
                    }
                } else {
                    found_leaves.push_back(node.n);
                    node_queue.push({node.n, cp});
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
    }

    ec.l.cs = ec_labels;

    D_COUT << "----------------------------------------------------------------------------------------------------\n";
}


// other
//----------------------------------------------------------------------------------------------------------------------

void finish_example(vw& all, plt& p, example& ec){

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

void pass_end(plt& p){
    cout << "end of pass " << p.all->passes_complete << "\n";
}

template<bool use_threshold>
void finish(plt& p){

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

base_learner* plt_setup(vw& all) //learner setup
{
    if (missing_option<size_t, true>(all, "plt", "Use probabilistic label tree for multilabel with <k> labels"))
        return nullptr;
    new_options(all, "plt options")
            ("kary_tree", po::value<uint32_t>(), "tree in which each node has no more than k children")
            ("inner_threshold", po::value<float>(), "threshold for positive label (default 0.15)")
            ("p_at", po::value<uint32_t>(), "P@k (default 1)")
            ("positive_labels", "print all positive labels")
            ("top_k_labels", "print top-k labels")
            ("greedy", "greedy prediction");
    add_options(all);

    plt& data = calloc_or_throw<plt>();
    data.k = (uint32_t)all.vm["plt"].as<size_t>();
    data.inner_threshold = -1;
    data.positive_labels = false;
    data.top_k_labels = false;
    data.greedy = false;
    data.all = &all;

    data.precision = 0;
    data.predicted_number = 0;
    data.prediction_count = 0;
    data.p_at_k = 1;
    data.n_visited_nodes = 0;
    data.ec_count = 0;

    // plt parse options
    //------------------------------------------------------------------------------------------------------------------

    learner<plt> *l;

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
    data.ti = data.t - data.k;
    *(all.file_options) << " --kary_tree " << data.kary;


    if( all.vm.count("inner_threshold"))
        data.inner_threshold = all.vm["inner_threshold"].as<float>();

    if( all.vm.count("p_at") )
        data.p_at_k = all.vm["p_at"].as<uint32_t>();

    if( all.vm.count("positive_labels"))
        data.positive_labels = true;

    if( all.vm.count("top_k_labels"))
        data.top_k_labels = true;

    if( all.vm.count("greedy"))
        data.greedy = true;

    // init multiclass learner
    // -----------------------------------------------------------------------------------------------------------------

    if (data.inner_threshold >= 0) {
        l = &init_multiclass_learner(&data, setup_base(all), learn, predict<true, false>, all.p, data.t);
        l->set_finish(finish<true>);
    }
    else if(data.greedy){
        data.p_at_k = 1;
        data.precision_at_k.resize(data.p_at_k);
        l = &init_multiclass_learner(&data, setup_base(all), learn, predict<false, true>, all.p, data.t);
        l->set_finish(finish<false>);
    }
    else{
        data.precision_at_k.resize(data.p_at_k);
        l = &init_multiclass_learner(&data, setup_base(all), learn, predict<false, false>, all.p, data.t);
        l->set_finish(finish<false>);
    }

    data.nodes_t = calloc_or_throw<float>(data.t);
    for(size_t i = 0; i < data.t; ++i) data.nodes_t[i] = all.initial_t;

    // override parser
    //------------------------------------------------------------------------------------------------------------------

    all.p->lp = COST_SENSITIVE::cs_label;
    all.cost_sensitive = make_base(*l);

    all.holdout_set_off = true; // turn off stop based on holdout loss


    // log info & add some event handlers
    //------------------------------------------------------------------------------------------------------------------
    cout << "plt\n" << "k = " << data.k << "\ntree size = " << data.t << "\nkary_tree = " << data.kary
         << "\ninner_threshold = " << data.inner_threshold << endl;

    l->set_finish_example(finish_example);
    l->set_save_load(save_load_nodes);
    l->set_end_pass(pass_end);

    return all.cost_sensitive;
}