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


struct node{
    uint32_t base_predictor; //id of the base predictor
    uint32_t label;

    node* parent; // pointer to the parent node
    vector<node*> children; // pointers to the children nodes
    bool internal; // internal or leaf
    float p; // prediction value

    bool operator < (const node& r) const { return p < r.p; }
};

struct oplt {
    vw* all;

    size_t predictor_bits;
    size_t max_predictors;
    size_t base_predictors_count;

    node *tree_root;
    vector<node*> tree; // pointers to tree nodes
    unordered_map<uint32_t, node*> tree_leaves; // leaves map
    node *temp_node;

    float inner_threshold;  // inner threshold
    bool positive_labels;   // print positive labels
    uint32_t p_at_k;
    float precision;
    float predicted_number;
    vector<float> precision_at_k;
    size_t prediction_count;

    default_random_engine rng;
};


// debug helpers
//----------------------------------------------------------------------------------------------------------------------

void oplt_example_info(oplt& p, base_learner& base, example& ec){
    cout << "TAG: " << (ec.tag.size() ? std::string(ec.tag.begin()) : "-") << " FEATURES COUNT: " << ec.num_features
         << " LABELS COUNT: " << ec.l.cs.costs.size() << endl;

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
    cout << "TREE SIZE: " << p.tree.size() << " TREE LEAVES: " << p.tree_leaves.size() << "\nTREE:\n";
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

bool compare_node_ptr_func(const node* l, const node* r) { return (*l < *r); }

struct compare_node_ptr_functor{
    bool operator()(const node* l, const node* r) const { return (*l < *r); }
};

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
    p.tree_leaves[label] = n;
}

void copy_weights(oplt& p, uint32_t w1, uint32_t w2){
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
    p.tree_leaves = unordered_map<uint32_t, node*>();
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
                if (!n->internal) p.tree_leaves[n->label] = n;
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
    if(p.tree_leaves.size() == 0){
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

void predict(oplt& p, base_learner& base, example& ec);

inline void learn_node(node* n, base_learner& base, example& ec){
    D_COUT << "LEARN NODE: " << n->base_predictor << " LABEL: " << ec.l.simple.label << endl;
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
            if (p.tree_leaves.find(cl.class_index) == p.tree_leaves.end()){
                new_label(p, base, ec, cl.class_index);
            }
        }
        for (auto &cl : ec_labels.costs) {
            node *n = p.tree_leaves[cl.class_index];
            n_positive.insert(n);
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

    predict(p, base, ec);

    D_COUT << "----------------------------------------------------------------------------------------------------\n";
}


// predict
//----------------------------------------------------------------------------------------------------------------------

inline float predict_node(node *n, base_learner& base, example& ec){
    D_COUT << "PREDICT NODE: " << n->base_predictor << endl;

    ec.l.simple = {FLT_MAX, 0.f, 0.f};
    base.predict(ec, n->base_predictor);

    if(DEBUG) oplt_prediction_info(base, ec);

    return logit(ec.partial_prediction);
}

void predict(oplt& p, base_learner& base, example& ec){

    D_COUT << "PREDICT EXAMPLE\n";
    if(DEBUG) oplt_example_info(p, base, ec);

    ++p.prediction_count;

    COST_SENSITIVE::label ec_labels = ec.l.cs;

    // threshold prediction
    if (p.inner_threshold >= 0) {
        vector <node*> positive_labels;
        queue <node*> n_queue;

        p.tree_root->p = 1.0f;
        n_queue.push(p.tree_root);

        while (!n_queue.empty()) {
            node *n = n_queue.front(); // current node
            n_queue.pop();

            float cp = n->p * predict_node(n, base, ec);

            if (cp > p.inner_threshold) {
                if (n->internal) {
                    for (auto child : n->children) {
                        child->p = cp;
                        n_queue.push(child);
                    }
                } else {
                    n->p = cp;
                    positive_labels.push_back(n);
                }
            }
        }

        sort(positive_labels.rbegin(), positive_labels.rend(), compare_node_ptr_func);

        if (p.p_at_k > 0 && ec_labels.costs.size() > 0) {
            for (size_t i = 0; i < p.p_at_k && i < positive_labels.size(); ++i) {
                p.predicted_number += 1.0f;
                for (auto &cl : ec_labels.costs) {
                    if (positive_labels[i]->label == cl.class_index) {
                        p.precision += 1.0f;
                        break;
                    }
                }
            }
        }
    }

    // top-k predictions
    else{
        vector <node*> best_labels, found_leaves;
        priority_queue <node*, vector<node*>, compare_node_ptr_functor> n_queue;

        p.tree_root->p = 1.0f;
        n_queue.push(p.tree_root);

        while (!n_queue.empty()) {
            node *n = n_queue.top(); // current node
            n_queue.pop();

            if (find(found_leaves.begin(), found_leaves.end(), n) != found_leaves.end()) {
                best_labels.push_back(n);
                if (best_labels.size() >= p.p_at_k) break;
            }
            else {
                float cp = n->p * predict_node(n, base, ec);

                if (n->internal) {
                    for (auto child : n->children) {
                        child->p = cp;
                        n_queue.push(child);
                    }
                } else {
                    n->p = cp;
                    found_leaves.push_back(n);
                    n_queue.push(n);
                }
            }
        }

        vector <uint32_t> true_labels;
        for (auto &cl : ec_labels.costs) true_labels.push_back(cl.class_index);

        if (p.p_at_k > 0 && true_labels.size() > 0) {
            for (size_t i = 0; i < p.p_at_k; ++i) {
                if (find(true_labels.begin(), true_labels.end(), best_labels[i]->label) != true_labels.end())
                    p.precision_at_k[i] += 1.0f;
            }
        }
    }

    ec.l.cs = ec_labels;

    D_COUT << "----------------------------------------------------------------------------------------------------\n";
}


// other
//----------------------------------------------------------------------------------------------------------------------

void finish_example(vw& all, oplt& p, example& ec){

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

        all.sd->update(ec.test_only, 0.0f, ec.l.multi.weight, ec.num_features);
    }
    */

    //MULTICLASS::print_update_with_probability(all, ec, pred);
    VW::finish_example(all, &ec);
}

void pass_end(oplt& p){
    cout << "end of pass (epoch) " << p.all->passes_complete << "\n";
}

void finish(oplt& p){
    if (p.inner_threshold >= 0) {/// THRESHOLD PREDICTION
        if (p.predicted_number > 0) {
            cout << "Precision = " << p.precision / p.predicted_number << "\n";
        } else {
            cout << "Precision unknown - nothing predicted" << endl;
        }
    } else {/// TOP-k PREDICTION
        float correct = 0;
        for (size_t i = 0; i < p.p_at_k; ++i) {
            correct += p.precision_at_k[i];
            cout << "P@" << i + 1 << " = " << correct / (p.prediction_count * (i + 1)) << "\n";
        }/// TOP-k PREDICTION
    }

    for(auto n : p.tree) delete n;
    p.tree_leaves.~unordered_map();
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
    data.inner_threshold = -1;
    data.positive_labels = false;
    data.all = &all;

    data.precision = 0;
    data.predicted_number = 0;
    data.prediction_count = 0;
    data.p_at_k = 1;

    data.rng.seed(time(0));

    init_tree(data);

    // oplt parse options
    // -----------------------------------------------------------------------------------------------------------------

    if (all.vm.count("inner_threshold"))
        data.inner_threshold = all.vm["inner_threshold"].as<float>();

    if( all.vm.count("positive_labels"))
        data.positive_labels = true;

    if( all.vm.count("p_at") )
        data.p_at_k = all.vm["p_at"].as<uint32_t>();

    if (data.inner_threshold < 0)
        data.precision_at_k.resize(data.p_at_k);

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

    l.set_finish_example(finish_example);
    l.set_save_load(save_load_tree);
    l.set_end_pass(pass_end);
    l.set_finish(finish);

    return all.cost_sensitive;
}
