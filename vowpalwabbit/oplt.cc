#include <float.h>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <queue>
#include <ctime>
#include <chrono>
#include <random>

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

struct node{
    uint32_t base_predictor; //id of the base predictor
    uint32_t label;

    node* parent; // pointer to the parent node
    vector<node*> children; // pointers to the children nodes
    bool internal; // internal or leaf
    float p; // prediction value
};

struct oplt {
    vw* all;

    size_t predictor_bits;
    size_t max_predictors;
    size_t base_predictors_count;

    node *tree_root;
    vector<node*> tree; // pointers to tree nodes
    unordered_map<uint32_t, node*> tree_leafs; // leafs map
    node *temp_node;

    float inner_threshold;  // inner threshold
    bool positive_labels;   // print positive labels

    uint32_t p_at_K;
    float p_at_P;
    size_t ec_count;

    default_random_engine rng;
};


// debug helpers
//----------------------------------------------------------------------------------------------------------------------

void oplt_example_info(oplt& p, base_learner& base, example& ec){
    cout << "PREDICT EXAMPLE: TAG: " << (ec.tag.size() ? std::string(ec.tag.begin()) : "-")
         << " FEATURES COUNT: " << ec.num_features << " LABELS COUNT: " << ec.l.cs.costs.size() << endl;

    cout << "BW: " << base.weights << " BI: " << base.increment
         << " WSS: " << p.all->weights.stride_shift() << " WM: " << p.all->weights.mask() << endl;

    for (features &fs : ec) {
        for (features::iterator_all &f : fs.values_indices_audit())
            cout << "FEATURE: " << (f.index() & p.all->weights.mask()) << " VALUE: " << f.value() << endl;
    }
    for (auto &cl : ec.l.cs.costs) cout << "LABEL: " << cl.class_index << endl;
}

void oplt_prediction_info(base_learner& base, example& ec){
    cout << std::fixed << std::setprecision(6) << "PP: " << ec.partial_prediction << " UP: " << ec.updated_prediction
         << " L: " << ec.loss << " S: " << ec.pred.scalar << endl;
}

void oplt_tree_info(oplt& p){
    cout << "TREE SIZE: " << p.tree.size() << " TREE LEAFS: " << p.tree_leafs.size() << "\nTREE:\n";
    queue<node*> n_queue;
    n_queue.push(p.tree_root);

    size_t depth = 0;
    while(!n_queue.empty()) {
        size_t q_size = n_queue.size();
        cout << "DEPTH " << depth << ": ";
        for(size_t i = 0; i < q_size; ++i){
            node *n = n_queue.front();
            n_queue.pop();

            if(n->parent) cout << "[" << n->parent->base_predictor << "]";
            cout << n->base_predictor;
            if(!n->internal) cout << "(" << n->label << ")";
            cout << " ";

            for(auto c : n->children) n_queue.push(c);
        }
        ++depth;
        cout << endl;
    }
}


// helpers
//----------------------------------------------------------------------------------------------------------------------

inline float logit(float in) { return 1.0f / (1.0f + exp(-in)); }

bool oplt_compare_label(const label &a, const label &b){
    return a.p > b.p;
}

node* init_node(oplt& p) {
    node* n = new node();
    n->base_predictor = p.base_predictors_count++;
    n->children.reserve(0);
    p.tree.push_back(n);
    n->internal = false;
    n->parent = nullptr;
    
    return n;
}

void node_set_parent(oplt& p, node *n, node *parent){
    n->parent = parent;
    parent->children.push_back(n);
    parent->internal = true;
}

void node_set_label(oplt& p, node *n, uint32_t label){
    n->internal = false;
    n->label = label;
    p.tree_leafs[label] = n;

    D_COUT << "TREE_MAP_DEBUG: " << p.tree_leafs.size() << endl;
}

void copy_weights(oplt& p, uint32_t w1, uint32_t w2){

    D_COUT << "COPY WEIGHTS\n";

    weight_parameters &weights = p.all->weights;
    uint64_t mask = weights.mask();
    uint64_t ws_size = mask >> p.predictor_bits;

    for (uint64_t i = 0; i <= ws_size; ++i) {
        uint64_t idx = (i << p.predictor_bits) & mask;
        weights[idx + w2] = weights[idx + w1];
    }
}

void init_tree(oplt& p){
    p.base_predictors_count = 0;
    p.tree_leafs = unordered_map<uint32_t, node*>();
    p.tree_root = init_node(p); // root node
    p.temp_node = init_node(p); // first temp node
}


// save/load
//----------------------------------------------------------------------------------------------------------------------

void save_load_tree(oplt& p, io_buf& model_file, bool read, bool text){

    D_COUT << "SAVE/LOAD TREE\n";

    if (model_file.files.size() > 0) {
        stringstream msg;

        // read/write predictor_bits
        bin_text_read_write_fixed(model_file, (char*)&p.predictor_bits, sizeof(p.predictor_bits), "", read, msg, text);
        msg << "predictor_bits = " << p.predictor_bits;

        // read/write number of predictors
        bin_text_read_write_fixed(model_file, (char*)&p.base_predictors_count, sizeof(p.base_predictors_count), "", read, msg, text);
        msg << " base_predictors_count = " << p.base_predictors_count;

        // read/write nodes
        size_t n_size;
        if(!read) n_size = p.tree.size();
        bin_text_read_write_fixed(model_file, (char*)&n_size, sizeof(n_size), "", read, msg, text);
        msg << " tree_size = " << n_size;

        if(read){
            for(size_t i = 0; i < n_size - 2; ++i) { // root and temp are already in tree after init
                node *n = new node();
                p.tree.push_back(n);
            }
        }

        // read/write root and temp nodes
        uint32_t root_predictor, temp_predictor;

        if(!read){
            root_predictor = p.tree_root->base_predictor;
            temp_predictor = p.temp_node->base_predictor;
        }

        bin_text_read_write_fixed(model_file, (char*)&root_predictor, sizeof(root_predictor), "", read, msg, text);
        bin_text_read_write_fixed(model_file, (char*)&temp_predictor, sizeof(temp_predictor), "", read, msg, text);

        // read/write base predictor, label
        for(auto n : p.tree) {
            bin_text_read_write_fixed(model_file, (char *) &n->base_predictor, sizeof(n->base_predictor), "", read, msg, text);
            bin_text_read_write_fixed(model_file, (char *) &n->label, sizeof(n->label), "", read, msg, text);
        }

        // read/write parent and rebuild tree
        for(auto n : p.tree) {
            uint32_t parent_base_predictor;

            if(!read){
                if(n->parent) parent_base_predictor = n->parent->base_predictor;
                else parent_base_predictor = -1;
            }

            bin_text_read_write_fixed(model_file, (char*)&parent_base_predictor, sizeof(parent_base_predictor), "", read, msg, text);

            if(read){
                if(n->base_predictor == root_predictor) p.tree_root = n;
                if(n->base_predictor == temp_predictor) p.temp_node = n;

                for (auto m : p.tree) {
                    if (m->base_predictor == parent_base_predictor) {
                        n->parent = m;
                        m->children.push_back(n);
                        break;
                    }
                }
            }
        }

        // recreate leafs index
        if(read) {
            for (auto n : p.tree) {
                n->internal = n->children.size();
                if (!n->internal) p.tree_leafs[n->label] = n;
            }
        }

        if(DEBUG) oplt_tree_info(p);
    }
}


// learn
//----------------------------------------------------------------------------------------------------------------------

node* expand_node(oplt& p, node* n, uint32_t new_label){
    D_COUT << "EXPAND NODE: BASE: " << n->base_predictor << " LABEL: " << n->label << " NEW LABEL: " << new_label << endl;

    node* copy_of_parent = init_node(p);
    node* new_node = p.temp_node;
    copy_weights(p, n->base_predictor, copy_of_parent->base_predictor);
    node_set_parent(p, copy_of_parent, n);
    node_set_label(p, copy_of_parent, n->label);
    node_set_parent(p, new_node, n);
    node_set_label(p, new_node, new_label);
    p.temp_node = init_node(p);

    n->children.shrink_to_fit();

    if(DEBUG) oplt_tree_info(p);

    return new_node;
}

node* new_label(oplt& p, base_learner& base, example& ec, uint32_t new_label){
    D_COUT << "NEW LABEL: " << new_label << endl;

    // if first label
    if(p.tree_leafs.size() == 0){
        node_set_label(p, p.tree_root, new_label);
        return p.tree_root;
    }

    node* to_expand = p.tree_root;
    size_t children_count = to_expand->children.size();

    // random policy
    uniform_int_distribution<uint32_t> bin_dist(0, children_count - 1);
    while(children_count){
        to_expand = to_expand->children[bin_dist(p.rng)];
        children_count = to_expand->children.size();
    }

    return expand_node(p, to_expand, new_label);
}

inline void learn_node(node* n, base_learner& base, example& ec){
    D_COUT << "LEARN NODE: " << n->base_predictor << endl;
    base.learn(ec, n->base_predictor);
    if(DEBUG) oplt_prediction_info(base, ec);
}

void learn(oplt& p, base_learner& base, example& ec){

    D_COUT << "LEARN EXAMPLE\n";
    if(DEBUG) oplt_example_info(p, base, ec);

    COST_SENSITIVE::label ec_labels = ec.l.cs;

    unordered_set<node*> n_positive; // positive nodes
    unordered_set<node*> n_negative; // negative nodes

    if (ec_labels.costs.size() > 0) {
        for (auto &cl : ec_labels.costs) {
            if (p.tree_leafs.find(cl.class_index) == p.tree_leafs.end()){
                new_label(p, base, ec, cl.class_index);
            }
        }
        for (auto &cl : ec_labels.costs) {
            node *n = p.tree_leafs[cl.class_index];
            while (n->parent) {
                n = n->parent;
                n_positive.insert(n);
            }
        }

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

    // learn positive and negative
    ec.l.simple = {1.f, 1.f, 0.f};
    for (auto &n : n_positive) learn_node(n, base, ec);

    ec.l.simple.label = -1.f;
    for (auto &n : n_negative) learn_node(n, base, ec);

    // learn temp node
    learn_node(p.temp_node, base, ec);

    ec.l.cs = ec_labels;
    ec.pred.multiclass = 0;

    D_COUT << "----------------------------------------------------------------------------------------------------\n";
}


// predict
//----------------------------------------------------------------------------------------------------------------------

inline void predict_node(node *n, base_learner& base, example& ec){
    D_COUT << "PREDICT NODE: " << n->base_predictor << endl;
    base.predict(ec, n->base_predictor);
    if(DEBUG) oplt_prediction_info(base, ec);
}

void predict(oplt& p, base_learner& base, example& ec){

    D_COUT << "PREDICT EXAMPLE\n";
    if(DEBUG) oplt_example_info(p, base, ec);

    COST_SENSITIVE::label ec_labels = ec.l.cs;

    v_array<float> ec_probs = v_init<float>();
    ec_probs.resize(p.tree_leafs.size());
    for(size_t i = 0; i < p.tree_leafs.size(); ++i) ec_probs[i] = 0.f;

    queue<node*> n_queue;

    p.tree_root->p = 1.0f;
    n_queue.push(p.tree_root);

    while(!n_queue.empty()) {
        node* n = n_queue.front(); // current node
        n_queue.pop();

        ec.l.simple = {FLT_MAX, 0.f, 0.f};
        predict_node(n, base, ec);
        float cp = n->p * logit(ec.partial_prediction);

        if(cp > p.inner_threshold) {
            if (n->internal) {
                for(auto child : n->children) {
                    child->p = cp;
                    n_queue.push(child);
                }
            }
            else{
                ec_probs[n->label - 1] = cp;
                D_COUT << " POSITIVE LABEL: " << n->label << ":" << cp << endl;
            }
        }
    }

    ec.pred.scalars = ec_probs;
    ec.l.cs = ec_labels;

    D_COUT << "----------------------------------------------------------------------------------------------------\n";
}


// other
//----------------------------------------------------------------------------------------------------------------------

void finish_example(vw& all, oplt& p, example& ec){

    D_COUT << "FINISH EXAMPLE\n";

    ++p.ec_count;
    vector<label> positive_labels;

    uint32_t pred = 0;
    for (uint32_t i = 0; i < p.tree_leafs.size(); ++i){
        if (ec.pred.scalars[i] > ec.pred.scalars[pred])
            pred = i;

        if (ec.pred.scalars[i] > p.inner_threshold)
            positive_labels.push_back({i + 1, ec.pred.scalars[i]});
    }
    ++pred; // prediction is {1..k} index (not 0)

    sort(positive_labels.begin(), positive_labels.end(), oplt_compare_label);

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

    for(auto n : p.tree) delete n;
    p.tree_leafs.~unordered_map();
}


// setup
//----------------------------------------------------------------------------------------------------------------------

base_learner* oplt_setup(vw& all) //learner setup
{
    if (missing_option<size_t, true>(all, "oplt", "Use online probabilistic label tree for multilabel with <k> labels"))
        return nullptr;
    new_options(all, "oplt options")
            ("inner_threshold", po::value<float>(), "threshold for positive label (default 0.15)")
            ("positive_labels", "print all positive labels")
            ("p_at", po::value<uint32_t>(), "P@k (default 1)");
    add_options(all);

    oplt& data = calloc_or_throw<oplt>();
    data.predictor_bits = all.vm["oplt"].as<size_t>();
    data.max_predictors = 1 << data.predictor_bits;
    data.inner_threshold = 0.15;
    data.positive_labels = false;
    data.all = &all;

    data.p_at_P = 0;
    data.ec_count = 0;
    data.p_at_K = 0;

    data.rng.seed(time(0));

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

    learner<oplt> &l = init_multiclass_learner(&data, setup_base(all), learn, predict, all.p, data.max_predictors);

    // override parser
    // -----------------------------------------------------------------------------------------------------------------

    all.p->lp = COST_SENSITIVE::cs_label;
    all.cost_sensitive = make_base(l);

    all.holdout_set_off = true; // turn off stop based on holdout loss

    // log info & add some event handlers
    // -----------------------------------------------------------------------------------------------------------------
    cout << "oplt\n" << "predictor_bits = " << data.predictor_bits << "\nmax_predictors = " << data.max_predictors
         << "\ninner_threshold = " << data.inner_threshold << endl;

    if(!all.training) {
        l.set_finish_example(finish_example);

    }

    l.set_save_load(save_load_tree);
    l.set_end_pass(pass_end);
    l.set_finish(finish);

    return all.cost_sensitive;
}
