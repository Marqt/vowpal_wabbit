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

#define DEBUG false
#define D_COUT if(DEBUG) cout

typedef struct{
    uint32_t l;
    float p;
} label;

struct node {
    uint32_t base_predictor; //id of the base predictor

    uint32_t id;
    uint32_t label;
    node* parent; // pointer to the parent node
    vector<node*> children; // pointers to the children nodes
    bool internal; // internal or leaf (faster then !leaf)

    float p; // prediction value
    uint32_t ec_count; // number of examples that reach this node
};

struct oplt {
    vw* all;

    uint32_t k; // number of labels
    size_t base_predictors_count;

    node *tree_root;
    vector<node*> tree; // pointers to tree nodes
    unordered_map<uint32_t, node*> tree_leafs; // leafs map

    float inner_threshold;  // inner threshold
    bool positive_labels;   // print positive labels

    uint32_t p_at_K;
    float p_at_P;
    size_t ec_count;
};


inline float logistic(float in) { return 1.0f / (1.0f + exp(-in)); }

bool compare_label2(const label &a, const label &b){
    return a.p > b.p;
}

inline void init_node(oplt& p, node* parent, uint32_t label = 0) {

    node* n = new node();

    n->base_predictor = p.base_predictors_count++;
    p.tree.push_back(n);
    n->parent = parent;
    n->ec_count = 0;
    n->p = 0.0;
    n->internal = true;

    if(label != 0) {
        n->internal = false;
        n->label = label;
        p.tree_leafs[label] = n;
    }

    n->parent->children.push_back(n);
}

void init_tree(oplt& p){
    // TODO: needs changes to be more elastic
    p.base_predictors_count = 1;

    uint32_t t = 2 * p.k - 1;

    node* root = new node();
    root->parent = nullptr;
    p.tree_root = root;
    p.tree.push_back(root);

    for(uint32_t i = 1; i < t; ++i) init_node(p, p.tree[floor((i - 1)/2)], i >= p.k - 1 ? i - p.k + 2 : 0);
}

void save_load_node(oplt& p, io_buf& model_file, bool read, bool text) {
    // TODO
}

void save_load_tree(oplt& p, io_buf& model_file, bool read, bool text){
    if (model_file.files.size() > 0) {
        stringstream msg;

        // read write k
        bin_text_read_write_fixed(model_file, (char*)&p.k, sizeof(p.k), "", read, msg, text);
        msg << "k = " << p.k;

        // read write number of predictors
        bin_text_read_write_fixed(model_file, (char*)&p.base_predictors_count, sizeof(p.base_predictors_count), "", read, msg, text);

        // read write nodes
        size_t n_size;
        if(!read) n_size = p.tree.size();
        bin_text_read_write_fixed(model_file, (char*)&n_size, sizeof(n_size), "", read, msg, text);

        // TODO
        // for(size_t i = 0; i < n_size; ++i) save_load_node(p, model_file, read, text);

        msg << "nodes = " << p.tree.size() << " ";
    }
}

inline void learn_node(node* n, base_learner& base, example& ec){
    ++n->ec_count;
    base.learn(ec, n->base_predictor);

    D_COUT << std::fixed << std::setw( 11 ) << std::setprecision( 6 ) << std::setfill( '0' ) << "LEARN NODE: " << n->base_predictor
           << " PP: " << ec.partial_prediction
           << " UP: " << ec.updated_prediction
           << " L: " << ec.loss
           << " S: " << ec.pred.scalar << endl;
}

void learn(oplt& p, base_learner& base, example& ec){

    D_COUT << "LEARN EXAMPLE: TAG: " << (ec.tag.size() ? std::string(ec.tag.begin()) : "-") << " F: " << ec.num_features << endl;

    COST_SENSITIVE::label ec_labels = ec.l.cs;

    unordered_set<node*> n_positive; // positive nodes
    unordered_set<node*> n_negative; // negative nodes

    if (ec_labels.costs.size() > 0) {
        for (auto& cl : ec_labels.costs) {
            D_COUT << " L: " << cl.class_index;
            if (cl.class_index > p.k)
                cout << "Label " << cl.class_index << " is not in {1," << p.k << "} This won't work right." << endl;

            node *n = p.tree_leafs[cl.class_index];
            while(n->parent){
                n = n->parent;
                n_positive.insert(n);
            }
        }

        D_COUT << endl;

        queue<node*> n_queue; // nodes queue
        n_queue.push(p.tree_root); // push root

        while(!n_queue.empty()) {
            node* n = n_queue.front(); // current node index
            n_queue.pop();

            if (n->internal) {
                for(auto child : n->children) {
                    if (n_positive.find(child) != n_positive.end()) n_queue.push(child);
                    else n_negative.insert(child);
                }
            }
        }
    }
    else
        n_negative.insert(p.tree_root);

    ec.l.simple = {1.f, 1.f, 0.f};
    for (auto &n : n_positive) learn_node(n, base, ec);

    ec.l.simple.label = -1.f;
    for (auto &n : n_negative) learn_node(n, base, ec);

    D_COUT << "----------------------------------------------------------------------------------------------------" << endl;

    ec.l.cs = ec_labels;
    ec.pred.multiclass = 0;
}

void predict(oplt& p, base_learner& base, example& ec){

    D_COUT << "PREDICT EXAMPLE: TAG: " << (ec.tag.size() ? std::string(ec.tag.begin()) : "-") << " F: " << ec.num_features << endl;

    COST_SENSITIVE::label ec_labels = ec.l.cs;

    if (DEBUG) {
        if (ec_labels.costs.size() > 0)
            for (auto &cl : ec_labels.costs) D_COUT << " L: " << cl.class_index;
        D_COUT << endl;
    }

    v_array<float> ec_probs;
    ec_probs.resize(p.k);
    for(int i = 0; i < p.k; ++i) ec_probs[i] = 0.f;

    queue<node*> n_queue;

    p.tree_root->p = 1.0f;
    n_queue.push(p.tree_root);

    while(!n_queue.empty()) {
        node* n = n_queue.front(); // current node
        n_queue.pop();

        ec.l.simple = {FLT_MAX, 0.f, 0.f};
        base.predict(ec, n->base_predictor);

        float cp = n->p * logistic(ec.partial_prediction);

        D_COUT << std::fixed << std::setw( 11 ) << std::setprecision( 6 ) << std::setfill( '0' ) << "PREDICT NODE: " << n->base_predictor
                << " NP: " << n->p
                << " PP: " << logistic(ec.partial_prediction)
                << " S: " << ec.pred.scalar << endl;

        if(cp > p.inner_threshold) {

            if (n->internal) {
                for(auto child : n->children) {
                    child->p = cp;
                    n_queue.push(child);
                }
            }
            else{
                ec_probs[n->label - 1] = cp;
                D_COUT << " PL: " << n->label << ":" << cp;
            }
        }
    }
    D_COUT << endl;

    ec.pred.scalars = ec_probs;
    ec.l.cs = ec_labels;
}

void finish_example(vw& all, oplt& p, example& ec){

    D_COUT << "FINISH EXAMPLE: TAG: " << (ec.tag.size() ? std::string(ec.tag.begin()) : "-") << " F: " << ec.num_features << endl;

    ++p.ec_count;
    vector<label> positive_labels;

    uint32_t pred = 0;
    for (uint32_t i = 0; i < p.k; ++i){
        if (ec.pred.scalars[i] > ec.pred.scalars[pred])
            pred = i;

        if (ec.pred.scalars[i] > p.inner_threshold)
            positive_labels.push_back({i + 1, ec.pred.scalars[i]});
    }
    ++pred; // prediction is {1..k} index (not 0)

    sort(positive_labels.begin(), positive_labels.end(), compare_label2);

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

    //MULTICLASS::print_update_with_probability(all, ec, pred);
    VW::finish_example(all, &ec);
}

void pass_end(oplt& p){
    cout << "end of pass (epoch) " << p.all->passes_complete << "\n";
}

void finish(oplt& p){
    if(p.p_at_K > 0)
        cout << "P@" << p.p_at_K << " = " << p.p_at_P / p.ec_count << "\n";
}

base_learner* plt_online_setup(vw& all) //learner setup
{
    if (missing_option<size_t, true>(all, "oplt", "Use online probabilistic label tree for multilabel with <k> labels"))
        return nullptr;
    new_options(all, "oplt options")
            ("inner_threshold", po::value<float>(), "threshold for positive label (default 0.15)")
            ("positive_labels", "print all positive labels")
            ("p_at", po::value<uint32_t>(), "P@k (default 1)");
    add_options(all);

    oplt& data = calloc_or_throw<oplt>();
    data.k = (uint32_t)all.vm["oplt"].as<size_t>();
    data.inner_threshold = 0.15;
    data.positive_labels = false;
    data.all = &all;

    data.p_at_P = 0;
    data.ec_count = 0;
    data.p_at_K = 0;

    init_tree(data);

    // oplt parse options
    // -----------------------------------------------------------------------------------------------------------------

    if (all.vm.count("inner_threshold"))
        data.inner_threshold = all.vm["inner_threshold"].as<float>();

    if( all.vm.count("positive_labels"))
        data.positive_labels = true;

    if( all.vm.count("p_at") )
        data.p_at_K = all.vm["p_at"].as<uint32_t>();

    // init learner
    // -----------------------------------------------------------------------------------------------------------------

    learner<oplt> &l = init_multiclass_learner(&data, setup_base(all), learn, predict, all.p, data.k);

    // override parser
    // -----------------------------------------------------------------------------------------------------------------

    all.p->lp = COST_SENSITIVE::cs_label;
    all.cost_sensitive = make_base(l);

    all.holdout_set_off = true; // turn off stop based on holdout loss

    // log info & add some event handlers
    // -----------------------------------------------------------------------------------------------------------------
    cout << "oplt\n" << "k = " << data.k << endl;
    cout << "inner_threshold = " << data.inner_threshold << endl;

    if(!all.training) {
        l.set_finish_example(finish_example);
        l.set_finish(finish);
    }
    l.set_end_pass(pass_end);

    return all.cost_sensitive;
}






