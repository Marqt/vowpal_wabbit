#include <float.h>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <fstream>
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
    uint32_t sub_depth;
    float t;
    float p; // prediction value
    bool operator < (const node& r) const { return p < r.p; }
};

struct oplt_alt {
    vw* all;

    size_t k;
    size_t predictor_bits;
    size_t max_predictors;
    size_t base_predictor_count;
    size_t kary;

    node *tree_root;
    node *subtree_root;
    vector<node*> tree; // pointers to tree nodes
    unordered_map<uint32_t, node*> tree_leaves; // leaves map

    float inner_threshold;  // inner threshold
    bool positive_labels;   // print positive labels
    bool top_k_labels;   // print top-k labels
    bool greedy;
    uint32_t p_at_k;
    float precision;
    float predicted_number;
    vector<float> precision_at_k;
    size_t prediction_count;

    // node expanding
    void(*copy)(oplt_alt& p, uint32_t wv1, uint32_t wv2);

    // stats
    uint32_t pass_count;
    uint32_t ec_count;
    uint32_t node_count;
    long n_visited_nodes;
    v_array<float> predictions;
    //struct timeb t_start, t_end;

    // save/load tree structure
    bool save_tree_structure;
    string save_tree_structure_file;
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
            if(n->temp) cout << "<" << n->temp->base_predictor << ">";
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

inline float sigmoid(float in) { return 1.0f / (1.0f + exp(-in)); }

bool alt_compare_node_ptr_func(const node* l, const node* r) { return (*l < *r); }

struct alt_compare_node_ptr_functor{
    bool operator()(const node* l, const node* r) const { return (*l < *r); }
};

inline node* init_node(oplt_alt& p) {
    node* n = new node();
    n->base_predictor = p.base_predictor_count++;
    n->children.reserve(0);
    n->internal = true;
    n->inverted = false;
    n->parent = nullptr;
    n->temp = nullptr;
    n->t = p.all->initial_t;
    n->sub_depth = 1;

    return n;
}

inline void node_set_parent(oplt_alt& p, node *n, node *parent){
    n->parent = parent;
    parent->children.push_back(n);
    parent->internal = true;
}

inline void node_set_label(oplt_alt& p, node *n, uint32_t label){
    n->internal = false;
    n->label = label;
    p.tree_leaves[label] = n;
}

inline node* node_copy(oplt_alt& p, node *n){
    node* c = init_node(p);
    p.copy(p, n->base_predictor, c->base_predictor);
    c->t = n->t;
    c->inverted = n->inverted;

    return c;
}

template<bool stride>
void copy_weights(oplt_alt& p, uint32_t wv1, uint32_t wv2){
    parameters &weights = p.all->weights;
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

inline void set_move_subtree(oplt_alt& p, node* n, node* m){
    if(n->internal) {
        for (auto child : n->children)
            node_set_parent(p, child, m);
        n->children.clear();
    }
    else
        node_set_label(p, m, n->label);

    node_set_parent(p, m, n);
}

inline uint32_t node_get_depth(oplt_alt& p, node* n){
    uint32_t n_depth = 1;

    while(n != p.tree_root){
        n = n->parent;
        ++n_depth;
    }

    return n_depth;
}

void init_tree(oplt_alt& p){
    p.base_predictor_count = 0;
    p.tree_leaves = unordered_map<uint32_t, node*>();
    p.tree_root = init_node(p); // root node
    p.tree_root->temp = init_node(p); // first temp node
    p.tree.push_back(p.tree_root);

    p.subtree_root = p.tree_root;
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
        msg << " base_predictor_count = " << p.base_predictor_count;
        bin_text_read_write_fixed(model_file, (char*)&p.base_predictor_count, sizeof(p.base_predictor_count), "", read, msg, text);

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

        // recreate leafs index
        if(read) {
            for (auto n : p.tree)
                if (!n->internal) p.tree_leaves[n->label] = n;
        }

        if(resume){
            uint32_t subtree_root_predictor;
            if(!read) subtree_root_predictor = p.subtree_root->base_predictor;
            bin_text_read_write_fixed(model_file, (char*)&subtree_root_predictor, sizeof(subtree_root_predictor), "", read, msg, text);

            for(auto n : p.tree) {
                if(read) if(n->base_predictor == subtree_root_predictor) p.subtree_root = n;
                bin_text_read_write_fixed(model_file, (char *) &n->t, sizeof(n->t), "", read, msg, text);
                bin_text_read_write_fixed(model_file, (char *) &n->sub_depth, sizeof(n->sub_depth), "", read, msg, text);
            }
        }

        if(DEBUG) oplt_alt_tree_info(p);
    }
}

vector<int> oplt_alt_parse_line(string text, char d = ' '){
    vector<int> result;
    const char *str = text.c_str();

    do {
        const char *begin = str;
        while(*str != d && *str) ++str;
        result.push_back(stoi(string(begin, str)));
    } while (0 != *str++);

    return result;
}

// load tree structure
void load_tree_structure(oplt_alt& p, string file_name){
    ifstream file;
    file.open(file_name);
    if(file.is_open()){
        string line;
        uint32_t line_count = 0;

        while(getline(file, line)) {
            ++line_count;
            if(!line.size() || line[0] == '#')
                continue;

            vector<int> nodes;

            try {
                nodes = oplt_alt_parse_line(line);
            }
            catch(...){
                cout << "Something is wrong with line " << line_count << " in " << file_name << "!\n";
                continue;
            };

            node* parent = nullptr;
            node* child = nullptr;
            for(auto n : p.tree){
                if(n->base_predictor == nodes[0]) parent = n;
                else if(n->base_predictor == nodes[1]) child = n;
            }
            if(!parent){
                parent = init_node(p);
                parent->base_predictor = nodes[0];
                p.tree.push_back(parent);
            }
            if(!child){
                child = init_node(p);
                child->base_predictor = nodes[1];
                p.tree.push_back(child);
            }
            node_set_parent(p, child, parent);

            if(nodes.size() >= 3){
                node_set_label(p, child, nodes[2]);
            }
        }

        p.k = p.tree_leaves.size();
        p.tree_root->temp = nullptr;

        file.close();
        cout << "Tree structure loaded from " << file_name << endl;
        oplt_alt_tree_info(p);
    }
}

// save tree structure
void save_tree_structure(oplt_alt& p, string file_name){
    ofstream file;
    file.open(file_name);
    if(file.is_open()){
        for(auto n : p.tree){
            for(auto c : n->children){
                file << n->base_predictor << " " << c->base_predictor;
                if(!c->internal) file << " " << c->label;
                file << endl;
            }
        }

        file.close();
        cout << "Tree structure saved to " << file_name << endl;
        oplt_alt_tree_info(p);
    }
}


// learn
//----------------------------------------------------------------------------------------------------------------------

void learn_node(oplt_alt& p, node* n, base_learner& base, example& ec){
    D_COUT << "LEARN NODE: " << n->base_predictor << " LABEL: " << ec.l.simple.label << " WEIGHT: " << ec.weight << " NODE_T: " << n->t << endl;

    p.all->sd->t = n->t;
    n->t += ec.weight;

    if(n->inverted){
        ec.l.simple.label *= -1.0f;
        base.learn(ec, n->base_predictor);
        ec.l.simple.label *= -1.0f;
    }
    else base.learn(ec, n->base_predictor);
    ++p.n_visited_nodes;
}

inline float predict_node(oplt_alt &p, node *n, base_learner& base, example& ec){
    D_COUT << "PREDICT NODE: " << n->base_predictor << endl;

    ec.l.simple = {FLT_MAX, 0.f, 0.f};
    base.predict(ec, n->base_predictor);

    if(n->inverted) ec.partial_prediction *= -1.0f;
    ++p.n_visited_nodes;

    return sigmoid(ec.partial_prediction);
}

node* add_new_label(oplt_alt& p, uint32_t new_label){
    D_COUT << "NEW LABEL: " << new_label << endl;

    if(p.tree_leaves.size() == 0) {
        node_set_label(p, p.tree_root, new_label);
        p.subtree_root = p.tree_root;
        return p.subtree_root;
    }

    if(p.tree.size() >= p.max_predictors){
        cerr << "Max number of nodes reached, tree can't be expanded.\n";
    }

    if(p.subtree_root->children.size() < p.kary){
        D_COUT << "ADDING NEW CHILD\n";

        if(!p.subtree_root->internal){
            node* label_child = node_copy(p, p.subtree_root->temp);
            node_set_parent(p, label_child, p.subtree_root);
            node_set_label(p, label_child, p.subtree_root->label);
            p.tree.push_back(label_child);

            ++p.subtree_root->sub_depth;
        }

        node* new_child;

        if(p.subtree_root->parent && p.subtree_root->children.size() == p.kary - 1 && p.subtree_root->parent->sub_depth == p.subtree_root->sub_depth + 1){
            new_child = p.subtree_root->temp;
            p.subtree_root->temp = nullptr;
        }
        else
            new_child = node_copy(p, p.subtree_root->temp);

        new_child->inverted = !new_child->inverted;
        node_set_parent(p, new_child, p.subtree_root);
        node_set_label(p, new_child, new_label);
        p.tree.push_back(new_child);

        if(p.subtree_root->sub_depth > 2) {
            p.subtree_root = new_child;
            new_child->temp = init_node(p);
        }

        return new_child;
    }
    else if(p.subtree_root->parent && p.subtree_root->parent->sub_depth == p.subtree_root->sub_depth + 1){
        D_COUT << "MOVING UP\n";
        p.subtree_root = p.subtree_root->parent;
        return add_new_label(p, new_label);
    }
    else{
        D_COUT << "EXPANDING SUBROOT\n";

        node *parent_of_old_tree = node_copy(p, p.subtree_root->temp);
        parent_of_old_tree->sub_depth = p.subtree_root->sub_depth;
        set_move_subtree(p, p.subtree_root, parent_of_old_tree);
        p.tree.push_back(parent_of_old_tree);

        ++p.subtree_root->sub_depth;

        return add_new_label(p, new_label);

//        node *parent_of_new_tree;
//
//        if(p.subtree_root->parent && p.subtree_root->parent->sub_depth == p.subtree_root->sub_depth + 1){
//            parent_of_new_tree = p.subtree_root->temp;
//            p.subtree_root->temp = nullptr;
//        }
//        else
//            parent_of_new_tree = node_copy(p, p.subtree_root->temp);
//
//        parent_of_new_tree->inverted = !parent_of_new_tree->inverted;
//        node_set_parent(p, parent_of_new_tree, p.subtree_root);
//        node_set_label(p, parent_of_new_tree, new_label);
//        p.tree.push_back(parent_of_new_tree);
//
//        if(node_get_depth(p, p.subtree_root) + 1 != p.tree_root->sub_depth) {
//            p.subtree_root = parent_of_new_tree;
//            parent_of_new_tree->temp = init_node(p);
//        }
//
//        return parent_of_new_tree;
    }
}

void learn(oplt_alt& p, base_learner& base, example& ec){

    D_COUT << "LEARN EXAMPLE: " << p.ec_count << " PASS: " << p.pass_count << "\n";
    if(DEBUG) oplt_alt_example_info(p, base, ec);

    COST_SENSITIVE::label ec_labels = ec.l.cs;
    double t = p.all->sd->t;
    double weighted_holdout_examples = p.all->sd->weighted_holdout_examples;
    p.all->sd->weighted_holdout_examples = 0;

    unordered_set<node*> n_positive; // positive nodes
    unordered_set<node*> n_negative; // negative nodes

    if (ec_labels.costs.size() > 0) {
        if(p.pass_count == 0) {
            for (auto &cl : ec_labels.costs) {
                if (p.tree_leaves.find(cl.class_index) == p.tree_leaves.end()) {
                    add_new_label(p, cl.class_index);
                    if (DEBUG) oplt_alt_tree_info(p);
                }
            }
        }

        for (auto &cl : ec_labels.costs) {
            node *n = p.tree_leaves[cl.class_index];
            n_positive.insert(n);
            if(n->temp) n_positive.insert(n->temp);

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
                if(n->temp) n_positive.insert(n->temp);
            }
        }
    }
    else
        n_negative.insert(p.tree_root);

    // learn positive and negative
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

template<bool use_threshold, bool greedy>
void predict(oplt_alt& p, base_learner& base, example& ec){

    D_COUT << "PREDICT EXAMPLE\n";
    if(DEBUG) oplt_alt_example_info(p, base, ec);

    if(p.prediction_count == 0) oplt_alt_tree_info(p);

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

    else if (greedy) {
        node* current = p.tree_root;
        current->p = predict_node(p, current, base, ec);

        while(current->internal){
            node* best = current->children[0];

            for (auto child : current->children) {
                child->p = current->p * predict_node(p, child, base, ec);
                if(best->p < child->p) best = child;
            }

            current = best;
        }

        vector<uint32_t> true_labels;
        for (auto &cl : ec_labels.costs) true_labels.push_back(cl.class_index);

        if (find(true_labels.begin(), true_labels.end(), current->label) != true_labels.end())
            p.precision_at_k[0] += 1.0f;
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

    all.sd->update(ec.test_only, 0, ec.weight, ec.num_features);
    //MULTICLASS::print_update_with_probability(all, ec, pred);

    VW::finish_example(all, &ec);
}

void pass_end(oplt_alt& p){
    if(p.pass_count == 0){
        for(auto n : p.tree) n->temp = nullptr;
    }

    ++p.pass_count;
    cerr << "end of pass " << p.pass_count << endl;
}

template<bool use_threshold>
void finish(oplt_alt& p){
    // threshold prediction

    if(p.save_tree_structure) save_tree_structure(p, p.save_tree_structure_file);
    oplt_alt_tree_info(p);

    if (use_threshold) {
        if (p.predicted_number > 0) {
            cerr << "Precision = " << p.precision / p.predicted_number << "\n";
        } else {
            cerr << "Precision unknown - nothing predicted" << endl;
        }
    }

    // top-k predictions
    else {
        float correct = 0;
        for (size_t i = 0; i < p.p_at_k; ++i) {
            correct += p.precision_at_k[i];
            cerr << "P@" << i + 1 << " = " << correct / (p.prediction_count * (i + 1)) << "\n";
        }
    }

    cerr << "visited nodes = " << p.n_visited_nodes << endl;
    cerr << "tree_size = " << p.tree.size() << endl;
    cerr << "base_predictor_count = " << p.base_predictor_count << endl;

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
            ("inner_threshold", po::value<float>(), "threshold for positive label (default 0.15)")
            ("p_at", po::value<uint32_t>(), "P@k (default 1)")
            ("positive_labels", "print all positive labels")
            ("save_tree_structure", po::value<string>(), "save tree structure to file")
            ("load_tree_structure", po::value<string>(), "load tree structure from file")
            ("greedy", "greedy prediction");
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

    data.n_visited_nodes = 0;
    data.ec_count = data.node_count = 0;
    data.pass_count = 0;


    // oplt_alt parse options
    // -----------------------------------------------------------------------------------------------------------------

    learner<oplt_alt> *l;

    // kary options
    if(all.vm.count("kary_tree"))
        data.kary = all.vm["kary_tree"].as<uint32_t>();
    *(all.file_options) << " --kary_tree " << data.kary;

    size_t k_left = 1;
    while(data.k > k_left) k_left *= 2;
    k_left /= 2;
    size_t k_right = data.k - k_left;

    if(data.kary > 2){
        double a = pow(data.kary, floor(log(data.k) / log(data.kary)));
        double b = data.k - a;
        double c = ceil(b / (data.kary - 1.0));
        double d = (data.kary * a - 1.0)/(data.kary - 1.0);
        double e = data.k - (a - c);
        data.max_predictors = static_cast<uint32_t>(e + d);
    }
    else
        data.max_predictors = 2 * data.k - 1;

    cout << data.max_predictors << endl;

    size_t max_depth = 1;
    size_t _nodes = 1;
    while(data.k > _nodes){
        _nodes *= data.kary;
        ++max_depth;
    };
    data.max_predictors += max_depth + 1;

    data.predictor_bits = static_cast<size_t>(floor(log2(data.max_predictors))) + 1;


    if (all.vm.count("inner_threshold"))
        data.inner_threshold = all.vm["inner_threshold"].as<float>();

    if( all.vm.count("p_at") )
        data.p_at_k = all.vm["p_at"].as<uint32_t>();

    if( all.vm.count("positive_labels"))
        data.positive_labels = true;

    if( all.vm.count("greedy"))
        data.greedy = true;

    if (data.inner_threshold >= 0) {
        l = &init_multiclass_learner(&data, setup_base(all), learn, predict<true, false>, all.p, data.max_predictors);
        l->set_finish(finish<true>);
    }
    else if(data.greedy){
        data.p_at_k = 1;
        data.precision_at_k.resize(data.p_at_k);
        l = &init_multiclass_learner(&data, setup_base(all), learn, predict<false, true>, all.p, data.max_predictors);
        l->set_finish(finish<false>);
    }
    else{
        data.precision_at_k.resize(data.p_at_k);
        l = &init_multiclass_learner(&data, setup_base(all), learn, predict<false, false>, all.p, data.max_predictors);
        l->set_finish(finish<false>);
    }

    if(all.weights.stride_shift())
        data.copy = copy_weights<false>;
    else
        data.copy = copy_weights<true>;


    // init tree
    // -----------------------------------------------------------------------------------------------------------------

    init_tree(data);

    if(all.vm.count("save_tree_structure")) {
        data.save_tree_structure = true;
        data.save_tree_structure_file = all.vm["save_tree_structure"].as<string>();
    }
    else
        data.save_tree_structure = false;
    if(all.vm.count("load_tree_structure"))
        load_tree_structure(data, all.vm["load_tree_structure"].as<string>());

    // override parser
    // -----------------------------------------------------------------------------------------------------------------

    all.p->lp = COST_SENSITIVE::cs_label;
    all.cost_sensitive = make_base(*l);

    all.holdout_set_off = true; // turn off stop based on holdout loss


    // log info & add some event handlers
    // -----------------------------------------------------------------------------------------------------------------

    cerr << "oplt_alt\n" << "predictor_bits = " << data.predictor_bits << "\nmax_predictors = " << data.max_predictors
    << "\nkary_tree = " << data.kary << "\ninner_threshold = " << data.inner_threshold << endl;

    l->set_finish_example(finish_example);
    l->set_save_load(save_load_tree);
    l->set_end_pass(pass_end);

    return all.cost_sensitive;
}
