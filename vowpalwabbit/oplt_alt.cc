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
#include <cstring>

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
    node* temp;
    vector<node*> children; // pointers to the children nodes
    bool internal; // internal or leaf
    bool inverted;
    uint32_t ec_count;
    float p; // prediction value
    bool operator < (const node& r) const { return p < r.p; }
};

struct oplt_alt {
    vw* all;

    size_t k;
    size_t predictor_bits;
    size_t max_predictors;
    size_t base_predictors_count;
    size_t kary;

    node *tree_root;
    node *temp_tree_root;
    vector<node*> tree; // pointers to tree nodes
    unordered_map<uint32_t, node*> tree_leaves; // leaves map
    size_t tree_depth;
    size_t temp_tree_depth;

    float inner_threshold;  // inner threshold
    bool positive_labels;   // print positive labels
    uint32_t p_at_k;
    float precision;
    float predicted_number;
    vector<float> precision_at_k;
    size_t prediction_count;
    float base_eta;
    uint32_t decay_step;

    void(*copy)(oplt_alt& p, uint32_t wv1, uint32_t wv2);
    node*(*new_label)(oplt_alt& p, base_learner& base, example& ec, uint32_t new_label);
    void(*learn_node)(oplt_alt& p, node* n, base_learner& base, example& ec);

    default_random_engine rng;
};


// debug helpers - to delete
//----------------------------------------------------------------------------------------------------------------------

void oplt_alt_example_info(oplt_alt& p, base_learner& base, example& ec){
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

void oplt_alt_prediction_info(oplt_alt& p, base_learner& base, example& ec){
    cout << std::fixed << std::setprecision(6) << "PP: " << ec.partial_prediction << " UP: " << ec.updated_prediction
    << " L: " << ec.loss << " S: " << ec.pred.scalar << " ETA: " << p.all->eta << endl;
}

void oplt_alt_print_all_weights(oplt_alt &p){
    cout << endl << "WEIGHTS:";
    for (uint64_t i = 0; i <= p.all->weights.mask(); ++i) {
        cout << " " << p.all->weights.first()[i];
        if(!((i + 1) % (int)pow(2, p.predictor_bits + p.all->weights.stride_shift()))) cout << " | " << endl;
    }
    cout << endl;
}

void oplt_alt_tree_info(oplt_alt& p){
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

bool alt_compare_node_ptr_func(const node* l, const node* r) { return (*l < *r); }

struct alt_compare_node_ptr_functor{
    bool operator()(const node* l, const node* r) const { return (*l < *r); }
};

node* init_node(oplt_alt& p) {
    node* n = new node();
    n->base_predictor = p.base_predictors_count++;
    n->children.reserve(0);
    //n->children.reserve(p.kary);
    n->internal = true;
    n->inverted = false;
    n->parent = nullptr;
    n->temp = nullptr;
    n->ec_count = 0;
    p.tree.push_back(n);

    return n;
}

void node_set_parent(oplt_alt& p, node *n, node *parent){
    n->parent = parent;
    parent->children.push_back(n);
    parent->internal = true;
}

void node_set_label(oplt_alt& p, node *n, uint32_t label){
    n->internal = false;
    n->label = label;
    p.tree_leaves[label] = n;
}

node* node_copy(oplt_alt& p, node *n){
    node* c = init_node(p);
    p.copy(p, n->base_predictor, c->base_predictor);
    c->ec_count = n->ec_count;
    c->inverted = n->inverted;

    return c;
}

template<bool stride>
void copy_weights(oplt_alt& p, uint32_t wv1, uint32_t wv2){
    weight_parameters &weights = p.all->weights;
    uint64_t mask = weights.mask();

    if(stride){

        uint32_t stride_shift = weights.stride_shift();
        uint32_t mask_shift = p.predictor_bits + stride_shift;
        size_t stride_size = (1 << stride_shift) * sizeof(uint32_t);
        uint64_t wv_count = mask >> mask_shift;

        wv1 = wv1 << stride_shift;
        wv2 = wv2 << stride_shift;

        for (uint64_t i = 0; i <= wv_count; ++i) {
            uint64_t idx = (i << mask_shift); //& mask;
            memcpy(&weights[idx + wv2], &weights[idx + wv1], stride_size);
        }
    }
    else{

        uint64_t wv_count = mask >> p.predictor_bits;

        for (uint64_t i = 0; i <= wv_count; ++i) {
            uint64_t idx = (i << p.predictor_bits); //& mask;
            weights[idx + wv2] = weights[idx + wv1];
        }
    }

}

void init_tree(oplt_alt& p){
    p.base_predictors_count = 0;
    p.tree_leaves = unordered_map<uint32_t, node*>();
    p.tree_root = init_node(p); // root node
    p.temp_tree_root = p.tree_root;
    p.tree_depth = 0;
    p.temp_tree_depth = p.tree_depth;
}


// save/load
//----------------------------------------------------------------------------------------------------------------------

void save_load_tree(oplt_alt& p, io_buf& model_file, bool read, bool text){

    D_COUT << "SAVE/LOAD TREE\n";

    if (model_file.files.size() > 0) {
        bool resume = p.all->save_resume;
        stringstream msg;
        msg << ":" << resume << "\n";
        bin_text_read_write_fixed(model_file, (char*) &resume, sizeof(resume), "", read, msg, text);

        // read/write predictor_bits
        msg << "predictor_bits = " << p.predictor_bits;
        bin_text_read_write_fixed(model_file, (char*)&p.predictor_bits, sizeof(p.predictor_bits), "", read, msg, text);

        // read/write number of predictors
        msg << " base_predictors_count = " << p.base_predictors_count;
        bin_text_read_write_fixed(model_file, (char*)&p.base_predictors_count, sizeof(p.base_predictors_count), "", read, msg, text);

        // read/write nodes
        size_t n_size;
        if(!read) n_size = p.tree.size();
        bin_text_read_write_fixed(model_file, (char*)&n_size, sizeof(n_size), "", read, msg, text);
        msg << " tree_size = " << n_size;

        if(read){
            for(size_t i = 0; i < n_size - 1; ++i) { // root and temp are already in tree after init
                node *n = new node();
                p.tree.push_back(n);
            }
        }

        // read/write root and temp nodes
        uint32_t root_predictor;
        if(!read) root_predictor = p.tree_root->base_predictor;
        bin_text_read_write_fixed(model_file, (char*)&root_predictor, sizeof(root_predictor), "", read, msg, text);

        // read/write base predictor, label
        for(auto n : p.tree) {
            bin_text_read_write_fixed(model_file, (char *) &n->base_predictor, sizeof(n->base_predictor), "", read, msg, text);
            bin_text_read_write_fixed(model_file, (char *) &n->label, sizeof(n->label), "", read, msg, text);
            bin_text_read_write_fixed(model_file, (char *) &n->inverted, sizeof(n->inverted), "", read, msg, text);
            bin_text_read_write_fixed(model_file, (char *) &n->internal, sizeof(n->internal), "", read, msg, text);
        }

        // read/write parent and rebuild tree
        for(auto n : p.tree) {
            uint32_t parent_base_predictor, temp_base_predictor;

            if(!read){
                if(n->parent) parent_base_predictor = n->parent->base_predictor;
                else parent_base_predictor = -1;
                if(n->temp) temp_base_predictor = n->temp->base_predictor;
                else temp_base_predictor = -1;
            }

            bin_text_read_write_fixed(model_file, (char*)&parent_base_predictor, sizeof(parent_base_predictor), "", read, msg, text);
            bin_text_read_write_fixed(model_file, (char*)&temp_base_predictor, sizeof(temp_base_predictor), "", read, msg, text);

            D_COUT << "NODE: BASE: " << n->base_predictor << " PARENT: " << parent_base_predictor << " TEMP: " << temp_base_predictor << endl;

            if(read){
                if(n->base_predictor == root_predictor) p.tree_root = n;

                for (auto m : p.tree) {
                    if (m->base_predictor == parent_base_predictor) {
                        n->parent = m;
                        m->children.push_back(n);
                    }
                    else if (m->base_predictor == temp_base_predictor) {
                        m->temp = n;
                    }
                }
            }
        }

        D_COUT << "TEST5\n";

        // recreate leafs index
        if(read) {
            for (auto n : p.tree) {
                if (!n->internal) p.tree_leaves[n->label] = n;
            }
        }

        if(resume){
            bin_text_read_write_fixed(model_file, (char *) &p.base_eta, sizeof(p.base_eta), "", read, msg, text);
            bin_text_read_write_fixed(model_file, (char *) &p.decay_step, sizeof(p.decay_step), "", read, msg, text);
            bin_text_read_write_fixed(model_file, (char *) &p.tree_depth, sizeof(p.tree_depth), "", read, msg, text);
            bin_text_read_write_fixed(model_file, (char *) &p.temp_tree_depth, sizeof(p.temp_tree_depth), "", read, msg, text);

            uint32_t temp_root_predictor;
            if(!read) temp_root_predictor = p.temp_tree_root->base_predictor;
            bin_text_read_write_fixed(model_file, (char*)&temp_root_predictor, sizeof(temp_root_predictor), "", read, msg, text);

            for(auto n : p.tree) {
                if(read) if(n->base_predictor == temp_root_predictor) p.temp_tree_root = n;
                bin_text_read_write_fixed(model_file, (char *) &n->ec_count, sizeof(n->ec_count), "", read, msg, text);
            }
        }

        if(DEBUG) oplt_alt_tree_info(p);
    }
}


// learn
//----------------------------------------------------------------------------------------------------------------------

template<bool t_decay, bool exp_decay, bool step_decay>
void learn_node(oplt_alt& p, node* n, base_learner& base, example& ec){
    D_COUT << "LEARN NODE: " << n->base_predictor << " LABEL: " << ec.l.simple.label << endl;

    if(t_decay) p.all->eta = p.base_eta / (1.0f + p.all->eta_decay_rate * n->ec_count++);
    if(exp_decay) p.all->eta = p.base_eta * exp(-p.all->eta_decay_rate * n->ec_count++);
    if(step_decay) p.all->eta = p.base_eta * pow(p.all->eta_decay_rate, floor(n->ec_count++/p.decay_step));

    if(n->inverted) ec.l.simple.label *= -1.0f;

    base.learn(ec, n->base_predictor);

    if(n->inverted) ec.l.simple.label *= -1.0f;

    if(DEBUG) oplt_alt_prediction_info(p, base, ec);
}

inline float predict_node(oplt_alt &p, node *n, base_learner& base, example& ec){
    D_COUT << "PREDICT NODE: " << n->base_predictor << endl;

    ec.l.simple = {FLT_MAX, 0.f, 0.f};
    base.predict(ec, n->base_predictor);

    if(DEBUG) oplt_alt_prediction_info(p, base, ec);

    if(n->inverted) ec.partial_prediction *= -1.0f;

    return logit(ec.partial_prediction);
}



node* expand_node(oplt_alt& p, node* n, uint32_t new_label){
    D_COUT << "EXPAND NODE: BASE: " << n->base_predictor << " NEW LABEL: " << new_label << endl;
    D_COUT << "TREE: DEPTH: " << p.tree_depth << " TEMP_DEPTH: " << p.temp_tree_depth << endl;

    if(p.tree.size() >= p.max_predictors){
        cout << "Max number of nodes reached, tree can't be expanded.";
        return n;
    }

    if(n->children.size() >= p.kary || !n->internal) {

        if (p.tree_root == p.temp_tree_root) {
            D_COUT << "EXPANDING ROOT\n";

            node *parent_node = node_copy(p, n);
            node_set_parent(p, n, parent_node);
            p.tree_root = parent_node;

            parent_node->temp = node_copy(p, n);
            parent_node->temp->inverted = !parent_node->temp->inverted;

            p.tree_root = parent_node;
            p.temp_tree_root = parent_node;

            ++p.tree_depth;
            p.temp_tree_depth = p.tree_depth;

            return expand_node(p, p.temp_tree_root, new_label);

        } else {
            D_COUT << "EXPANDING UP\n";

            p.temp_tree_root = n->parent;
            ++p.temp_tree_depth;
            return expand_node(p, p.temp_tree_root, new_label);
        }
    }
    else {
        D_COUT << "EXPANDING DOWN\n";

        node *new_node;

        if (n->children.size() == p.kary - 1) {
            new_node = n->temp;
            n->temp = nullptr;
        } else
            new_node = node_copy(p, n->temp);

        node_set_parent(p, new_node, n);

        if(p.temp_tree_depth == 1) {
            node_set_label(p, new_node, new_label);
            return new_node;
        } else {
            new_node->temp = init_node(p);
            new_node->internal = true;

            p.temp_tree_root = new_node;
            --p.temp_tree_depth;

            return expand_node(p, new_node, new_label);
        }
    }
}

template<bool best_predicion>
node* new_label(oplt_alt& p, base_learner& base, example& ec, uint32_t new_label){
    D_COUT << "NEW LABEL: " << new_label << endl;

    // if first label
    if(p.tree_leaves.size() == 0){
        node_set_label(p, p.tree_root, new_label);
        return p.tree_root;
    }

    return expand_node(p, p.temp_tree_root, new_label);
}

void learn(oplt_alt& p, base_learner& base, example& ec){

    D_COUT << "LEARN EXAMPLE\n";
    if(DEBUG) oplt_alt_example_info(p, base, ec);

    COST_SENSITIVE::label ec_labels = ec.l.cs;

    unordered_set<node*> n_positive; // positive nodes
    unordered_set<node*> n_negative; // negative nodes

    if (ec_labels.costs.size() > 0) {
        for (auto &cl : ec_labels.costs) {
            if (p.tree_leaves.find(cl.class_index) == p.tree_leaves.end()){
                p.new_label(p, base, ec, cl.class_index);
                if(DEBUG) oplt_alt_tree_info(p);
            }
        }
        for (auto &cl : ec_labels.costs) {
            node *n = p.tree_leaves[cl.class_index];
            n_positive.insert(n);
            if(n->temp) n_negative.insert(n->temp);

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
                if(n->temp) n_negative.insert(n->temp);
            }
        }
    }
    else
        n_negative.insert(p.tree_root);

    // learn positive and negative
    ec.l.simple = {1.f, 1.f, 0.f};
    for (auto &n : n_positive) p.learn_node(p, n, base, ec);

    ec.l.simple.label = -1.f;
    for (auto &n : n_negative) p.learn_node(p, n, base, ec);

    if(DEBUG) oplt_alt_print_all_weights(p);

    ec.l.cs = ec_labels;
    ec.pred.multiclass = 0;

    D_COUT << "----------------------------------------------------------------------------------------------------\n";
}


// predict
//----------------------------------------------------------------------------------------------------------------------

template<bool use_threshold>
void predict(oplt_alt& p, base_learner& base, example& ec){

    D_COUT << "PREDICT EXAMPLE\n";
    if(DEBUG) oplt_alt_example_info(p, base, ec);

    ++p.prediction_count;

    COST_SENSITIVE::label ec_labels = ec.l.cs;

    // threshold prediction
    if (use_threshold) {
        vector<node*> positive_labels;
        queue<node*> n_queue;

        p.tree_root->p = 1.0f;
        n_queue.push(p.tree_root);

        while (!n_queue.empty()) {
            node *n = n_queue.front(); // current node
            n_queue.pop();

            float cp = n->p * predict_node(p, n, base, ec);

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

        sort(positive_labels.rbegin(), positive_labels.rend(), alt_compare_node_ptr_func);

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
        vector<node*> best_labels, found_leaves;
        priority_queue<node*, vector<node*>, alt_compare_node_ptr_functor> n_queue;

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
                float cp = n->p * predict_node(p, n, base, ec);

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

        vector<uint32_t> true_labels;
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

void finish_example(vw& all, oplt_alt& p, example& ec){

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

void pass_end(oplt_alt& p){
    cout << "end of pass " << p.all->passes_complete << "\n";
}

template<bool use_threshold>
void finish(oplt_alt& p){
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

    for(auto n : p.tree) delete n;
    p.tree_leaves.~unordered_map();
}

// setup
//----------------------------------------------------------------------------------------------------------------------

base_learner* oplt_alt_setup(vw& all) //learner setup
{
    if (missing_option<size_t, true>(all, "oplt_alt", "Use online probabilistic label tree for multilabel with max <k> labels"))
        return nullptr;
    new_options(all, "oplt_alt options")
            ("kary_tree", po::value<uint32_t>(), "tree in which each node has no more than k children")
            ("1t_decay", "eta = eta0 / (1 + decay * t)")
            ("exp_decay", "eta = eta0 * exp(-decay * t)")
            ("step_decay", po::value<uint32_t>(), "eta *= decay every step")
            ("random_policy", "expand random node")
            ("best_prediction_policy", "expand node with best prediction value")
            ("exp_decay", "eta = eta0 * exp(-decay * t)")
            ("inner_threshold", po::value<float>(), "threshold for positive label (default 0.15)")
            ("p_at", po::value<uint32_t>(), "P@k (default 1)")
            ("positive_labels", "print all positive labels");
    add_options(all);

    oplt_alt& data = calloc_or_throw<oplt_alt>();
    data.k = all.vm["oplt_alt"].as<size_t>();
    data.kary = 2;
    data.inner_threshold = -1;
    data.positive_labels = false;
    data.all = &all;

    data.precision = 0;
    data.predicted_number = 0;
    data.prediction_count = 0;
    data.p_at_k = 1;

    data.base_eta = all.eta;

    data.rng.seed(time(0));

    init_tree(data);

    // oplt_alt parse options
    // -----------------------------------------------------------------------------------------------------------------

    learner<oplt_alt> *l;

    string decay_type = "vw_decay";
    string expand_policy = "random_policy";

    // kary options
    if(all.vm.count("kary_tree"))
        data.kary = all.vm["kary_tree"].as<uint32_t>();
    *(all.file_options) << " --kary_tree " << data.kary;

    data.max_predictors = 1;
    while(data.k > data.max_predictors)
        data.max_predictors *= data.kary;
    data.predictor_bits = static_cast<size_t>(floor(log2(data.max_predictors))) + 1;

    // decay policy options
    if(all.vm.count("1t_decay")) {
        data.learn_node = learn_node<true, false, false>;
        decay_type = "1t_decay";
    }
    else if(all.vm.count("exp_decay")) {
        data.learn_node = learn_node<false, true, false>;
        decay_type = "exp_decay";
    }
    else if(all.vm.count("step_decay")) {
        data.decay_step = all.vm["step_decay"].as<uint32_t>();
        data.learn_node = learn_node<false, false, true>;
        decay_type = "step_decay = " + to_string(data.decay_step);
    }
    else
        data.learn_node = learn_node<false, false, false>;

    // expand policy options
    if(all.vm.count("random_policy"))
        data.new_label = new_label<false>;
    else if(all.vm.count("best_prediction_policy")) {
        data.new_label = new_label<true>;
        expand_policy = "best_prediction_policy";
    }
    else
        data.new_label = new_label<false>;

    if (all.vm.count("inner_threshold"))
        data.inner_threshold = all.vm["inner_threshold"].as<float>();

    if( all.vm.count("p_at") )
        data.p_at_k = all.vm["p_at"].as<uint32_t>();

    if( all.vm.count("positive_labels"))
        data.positive_labels = true;

    if (data.inner_threshold >= 0) { ;
        l = &init_multiclass_learner(&data, setup_base(all), learn, predict<true>, all.p, data.max_predictors);
        l->set_finish(finish<true>);
    }
    else{
        data.precision_at_k.resize(data.p_at_k);
        l = &init_multiclass_learner(&data, setup_base(all), learn, predict<false>, all.p, data.max_predictors);
        l->set_finish(finish<false>);
    }

    if(all.weights.stride_shift())
        data.copy = copy_weights<false>;
    else
        data.copy = copy_weights<true>;


    // override parser
    // -----------------------------------------------------------------------------------------------------------------

    all.p->lp = COST_SENSITIVE::cs_label;
    all.cost_sensitive = make_base(*l);

    all.holdout_set_off = true; // turn off stop based on holdout loss


    // log info & add some event handlers
    // -----------------------------------------------------------------------------------------------------------------
    cout << "oplt_alt\n" << "predictor_bits = " << data.predictor_bits << "\nmax_predictors = " << data.max_predictors
    << "\nkary_tree = " << data.kary << "\ninner_threshold = " << data.inner_threshold << endl;

    if(decay_type.length())
        cout << decay_type << endl;

    if(decay_type.length())
        cout << expand_policy << endl;

    // l.set_finish_example(finish_example);
    l->set_save_load(save_load_tree);
    l->set_end_pass(pass_end);

    return all.cost_sensitive;
}
