#include <float.h>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <queue>

#include "reductions.h"
#include "vw.h"

using namespace std;
using namespace LEARNER;


#define E_LOG true
#define LOG if(E_LOG) cout

#define E_D_LOG false
#define D_LOG if(E_D_LOG) cout

typedef struct{
    uint32_t n;
    float p;
} node;

typedef struct{
    uint32_t l;
    float p;
} label;

struct plt {
    vw* all;
    uint32_t k; // number of labels
    uint32_t t; // number of tree nodes
    float inner_threshold;  // inner threshold
    bool positive_labels;   // print positive labels
    uint32_t p_at_K;
    float p_at_P;
    size_t ec_count;
};

// y = f(x) -> [0, 1]

#define SIGMOID_TABLE_SIZE 512
#define MAX_SIGMOID 50
#define LOG_TABLE_SIZE 512

inline float logistic(float in) { return 1.0f / (1.0f + exp(-in)); }

void learn_node(plt& p, base_learner& base, example& ec, uint32_t n, bool positive) {

    D_LOG << "\nLEARN NODE: N: " << n << " P?: " << positive << " ";

    ec.l.simple = { positive ? 1.f : -1.f, 1.f, 0.f};
    base.learn(ec, n);

    D_LOG << " PP: " << ec.partial_prediction << " S: " << ec.pred.scalar << " P: " << ec.pred.prob;

}

void learn(plt& p, base_learner& base, example& ec){

    //D_LOG << "LEARN EXAMPLE: TAG: " << std::string(ec.tag.begin())  << " F: " << ec.num_features << " ";

    // multilabel - doesn't work?
    //MULTILABEL::labels multilabels = ec.l.multilabels;

    // multilabel with cost
    COST_SENSITIVE::label ec_labels = ec.l.cs; //example labels

    unordered_set<uint32_t> n_positive; // positive nodes
    unordered_set<uint32_t> n_negative; // negative nodes

    if (ec_labels.costs.size() > 0) {

        for (auto& cl : ec_labels.costs) {
            D_LOG << "L: " << cl.class_index << ":" << cl.partial_prediction << " ";
            // leaf index ( -2 because labels in {1..k})
            uint32_t tn = cl.class_index + p.k - 2;
            n_positive.insert(tn);
            while(tn > 0) {
                tn = floor((tn - 1)/2);
                n_positive.insert(tn);
            }
        }

        queue<uint32_t> n_queue; // nodes queue
        n_queue.push(0);

        while(!n_queue.empty()) {
            uint32_t n = n_queue.front(); // current node index
            n_queue.pop();

            if (n < p.k - 1) {
                uint32_t n_left_child = 2 * n + 1; // node left child index
                uint32_t n_right_child = 2 * n + 2; // node right child index

                bool n_left_child_positive
                        = n_positive.find(n_left_child) != n_positive.end();
                bool n_right_child_positive
                        = n_positive.find(n_right_child) != n_positive.end();

                if (n_left_child_positive) {
                    n_queue.push(n_left_child);
                    if (!n_right_child_positive)
                        n_negative.insert(n_right_child);
                }

                if (n_right_child_positive) {
                    n_queue.push(n_right_child);
                    if (!n_left_child_positive)
                        n_negative.insert(n_left_child);
                }
            }
        }
    }
    else
    {
        n_negative.insert(0);
    }

    for(auto& n : n_positive)
        learn_node(p, base, ec, n, true);

    for(auto& n : n_negative)
        learn_node(p, base, ec, n, false);

    ec.l.cs = ec_labels;
    ec.pred.multiclass = 0;

    D_LOG << "\nLEARN END\n";
}

void predict_node(plt& p, base_learner& base, example& ec, uint32_t n) {

    D_LOG << "\nPREDICT NODE: N: " << n << " ";
    ec.l.simple = {FLT_MAX, 0.f, 0.f};
    base.predict(ec, n);

    D_LOG << " PP: " << ec.partial_prediction << " S: " << ec.pred.scalar << " P: " << ec.pred.prob;

}

void predict(plt& p,  base_learner& base, example& ec){

    //D_LOG << "PREDICT EXAMPLE: TAG: " << std::string(ec.tag.begin())  << " ";

    COST_SENSITIVE::label ec_labels = ec.l.cs; //example's labels
    float *ec_probs = calloc_or_throw<float>(p.k);

    queue<node> node_queue;
    vector<label> label_positive;

    node_queue.push({0, 1.0f});

    while(!node_queue.empty()) {
        node node = node_queue.front(); // current node
        node_queue.pop();

        predict_node(p, base, ec, node.n);
        float cp = node.p * logistic(ec.pred.scalar);

        if(cp > p.inner_threshold) {

            if (node.n < p.k - 1) {
                uint32_t n_left_child = 2 * node.n + 1; // node left child index
                uint32_t n_right_child = 2 * node.n + 2; // node right child index
                node_queue.push({n_left_child, cp});
                node_queue.push({n_right_child, cp});
            }
            else{
                uint32_t l = node.n - p.k + 2;

                D_LOG << "\nPOSITIVE LABEL: "<< l << ":" << cp;

                label_positive.push_back({l, cp});
                ec_probs[l - 1] = cp;
            }
        }
    }

    if(p.positive_labels){
        ec.pred.probs = ec_probs;
    }

    ec.l.cs = ec_labels;

    D_LOG << "\nPREDINCT END\n";
}

bool compare_label(const label &a, const label &b){
    return a.p > b.p;
}

void finish_example(vw& all, plt& p, example& ec){

    ++p.ec_count;

    vector<label> positive_labels;

    uint32_t pred = 0;
    for (uint32_t i = 0; i < p.k; ++i){
        if (ec.pred.probs[i] > ec.pred.probs[pred])
            pred = i;

        if (ec.pred.probs[i] > p.inner_threshold)
            positive_labels.push_back({i + 1, ec.pred.probs[i]});
    }
    ++pred; // prediction is {1..k} index (not 0)

    sort(positive_labels.begin(), positive_labels.end(), compare_label);

    COST_SENSITIVE::label ec_labels = ec.l.cs; //example's labels

    if(p.p_at_K > 0 && ec_labels.costs.size() > 0) {
        for (size_t i = 0; i < p.p_at_K && i < positive_labels.size(); ++i) {
            for (auto &cl : ec_labels.costs) {
                if (positive_labels[i].l == cl.class_index) {
                    p.p_at_P += 1.0f / p.p_at_K;
                    break;
                }
            }
        }
    }

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

        all.sd->update(ec.test_only, 0.0f, ec.l.multi.weight, ec.num_features);
    }

    MULTICLASS::print_update_with_probability(all, ec, pred);
    free(ec.pred.probs);
    VW::finish_example(all, &ec);
}

void pass_end(plt& p){
    LOG << "end of pass (epoch) " << p.all->passes_complete << "\n";
}

void finish(plt& p){
    if(p.p_at_K > 0)
        LOG << "P@" << p.p_at_K << " = " << p.p_at_P / p.ec_count << "\n";
}

//void save_load(plt& p, io_buf& model_file, bool read, bool text){
//    LOG << "additional save/load\n";
//}
//
//void examples_end(plt& p){
//    LOG << "end of all examples\n";
//}

base_learner* plt_setup(vw& all) //learner setup
{
    if (missing_option<size_t, true>(all, "plt", "Use probabilistic label tree for multiclass/label with <k> labels"))
        return nullptr;
    new_options(all, "plt options")
            ("inner_threshold", po::value<float>(), "threshold for positive label (default 0.15)")
            ("positive_labels", "print all positive labels")
            ("p_at", po::value<uint32_t>(), "P@k (default 1)");
    add_options(all);

    plt& data = calloc_or_throw<plt>();
    data.k = (uint32_t)all.vm["plt"].as<size_t>();
    data.t = 2 * data.k - 1;
    data.inner_threshold = 0.15;
    data.positive_labels = false;
    data.all = &all;

    data.p_at_P = 0;
    data.ec_count = 0;
    data.p_at_K = 1;

    // plt parse options
    // ----------------------------------------

    if (all.vm.count("inner_threshold"))
        data.inner_threshold = all.vm["inner_threshold"].as<float>();

    if( all.vm.count("positive_labels"))
        data.positive_labels = true;

    if( all.vm.count("p_at") )
        data.p_at_K = all.vm["p_at"].as<uint32_t>();

    // init learner
    // ----------------------------------------

    learner<plt> &l = init_learner(&data, setup_base(all), learn, predict, data.t);

    // multiclass learner and parser
    //learner<plt> &l = init_multiclass_learner(&data, setup_base(all), learn, predict, all.p, data.k);


    // override default values of some params
    // ----------------------------------------
    // override parser type with multilabel with cost parser
    all.p->lp = COST_SENSITIVE::cs_label;
    all.cost_sensitive = make_base(l);

    // default learning rate to 10
    if(!all.vm.count("learning_rate") && !all.vm.count("l"))
        all.eta = 10;

    // default initial_t to 1 instead of 0
    if(!all.vm.count("initial_t")) {
        all.sd->t = 1.f;
        all.sd->weighted_unlabeled_examples = 1.f;
        all.initial_t = 1.f;
    }
    all.eta *= powf((float)(all.sd->t), all.power_t);
    all.holdout_set_off = true;


    // log info & add some event handlers
    // ----------------------------------------
    LOG << "plt\n" << "k = " << data.k << "\ntree size = " << data.t << endl;

    if(!all.training) {
        LOG << "inner_threshold = " << data.inner_threshold << "\n";
        l.set_finish_example(finish_example);
        l.set_finish(finish);
    }

    l.set_end_pass(pass_end);

    return all.cost_sensitive;
}


