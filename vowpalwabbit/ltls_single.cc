#include <float.h>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <queue>
#include <list>
#include <limits>
#include <chrono>
#include <random>

#include <boost/range/adaptor/reversed.hpp>

#include "reductions.h"
#include "vw.h"

using namespace std;
using namespace LEARNER;

#define CHECKS false
#define SKIPS_MUL false
#define SKIPS_FOR true
#define MULTILABEL false

#define DEBUG false
#define D_COUT if(DEBUG) cout

struct edge;
struct vertex;

struct edge{
    uint32_t e;
    vertex *v_out;
    vertex *v_in;
    uint32_t skips;

    float score;
    bool removed;
    float gradient;
    int positive;
};

#define MAX_E_OUT 3
struct vertex{
    uint32_t v;
    edge* E[MAX_E_OUT];

    edge* e_prev;
    bool removed;
    float score;
    float alpha;
    float beta;

    bool operator < (const vertex& r) const { return score < r.score; }
};

struct path{
    vector<edge*> E;
    float score;
    uint32_t label;

    bool operator < (const path& r) const { return score < r.score; }
};

// helpers
//----------------------------------------------------------------------------------------------------------------------

bool single_compare_path_ptr_func(const path* l, const path* r) { return (*l < *r); }
struct single_compare_path_ptr_functor{ bool operator()(const path* l, const path* r) const { return (*l < *r); }};

bool single_compare_vertex_ptr_func(const vertex* l, const vertex* r) { return (*l < *r); }
struct single_compare_vertex_ptr_functor{ bool operator()(const vertex* l, const vertex* r) const { return (*l < *r); }};

class ltls_loss_function : public loss_function {
public:
    float loss;
    float gradient;

    ltls_loss_function() { }

    float getLoss(shared_data*, float prediction, float label) {
        return loss;
    }

    float getUpdate(float prediction, float label, float update_scale, float pred_per_update) {
        return gradient * update_scale;
    }

    float getUnsafeUpdate(float prediction, float label, float update_scale) {
        return gradient * update_scale;
    }

    float first_derivative(shared_data*, float prediction, float label) {
        return gradient;
    }

    float second_derivative(shared_data*, float prediction, float label) {
        return 0;
    }

    float getRevertingWeight(shared_data*, float prediction, float eta_t) {
        return 0;
    }

    float getSquareGrad(float prediction, float label) {
        return gradient * gradient;
    }

};


struct ltls_single {
    vw* all;

    uint32_t k; // number of labels
    uint32_t e; // number of edges
    uint32_t v; // number of vertices
    vector<uint32_t> layers;

    vertex *v_source;
    vertex *v_sink;
    vector<vertex*> V; // vertices
    vector<edge*> E; // in topological order
    vector<path*> P; // paths

    uint32_t ranking_size;
    vector<path*> P_ranking;
    priority_queue<path*, vector<path*>, single_compare_path_ptr_functor> P_potential_ranking;

    uint32_t p_at_k;
    float precision;
    float predicted_number;
    vector<float> precision_at_k;
    size_t prediction_count;

    ltls_loss_function *loss_function;

};


// inits
//----------------------------------------------------------------------------------------------------------------------

edge* init_edge(ltls_single& l, vertex *v_out, vertex *v_in){
    if(!v_in || !v_out || v_in == v_out) return nullptr;

    edge *e = new edge();
    e->e = l.e++;
    e->v_out = v_out;
    e->v_in = v_in;
    e->skips = 1;
    e->removed = false;
    l.E.push_back(e);

    for(size_t i = 0; i < MAX_E_OUT; ++i){
        if(!e->v_out->E[i]){
            e->v_out->E[i] = e;
            break;
        }
    }

    return e;
}

vertex* init_vertex(ltls_single& l){
    vertex *v = new vertex();
    v->v = l.v++;
    v->E[0] = nullptr;
    v->E[1] = nullptr;
    v->E[2] = nullptr;
    l.V.push_back(v);
    return v;
}

void init_paths(ltls_single& l){

    for(size_t label = 1; label <= l.k; ++label) {
        D_COUT << "INIT PATH " << label << "\n";

        path *p = new path();
        p->score = 0;
        p->label = label;
        uint32_t _label, _layers;

        _label = label - 1;
        _layers = l.layers.size() - 1;
        while (_label >= l.layers[_layers]) {
            _label -= l.layers[_layers];
            --_layers;
        }

        vertex *_v = l.v_source;
        edge *_e;

        for (size_t i = 0; i <= _layers; ++i) {
            _e = _v->E[_label % 2];
            _label = _label >> 1;
            p->E.push_back(_e);
            _v = _e->v_in;
        }

        if (_v != l.v_sink) {
            for (size_t i = MAX_E_OUT - 1; i >= 0; --i) {
                if (_v->E[i] && _v->E[i]->v_in == l.v_sink) {
                    p->E.push_back(_v->E[i]);
                    break;
                }
            }
        }

        l.P.push_back(p);
    }
}

void init_graph(ltls_single& l){
    D_COUT << "INIT GRAPH\n";

    l.e = l.v = 0;
    l.v_source = init_vertex(l);
    l.v_sink = init_vertex(l);
    l.V.pop_back();

    // calc layers
    size_t i = 0;
    uint32_t _k = l.k;
    while(_k > 0){
        l.layers.push_back((_k % 2) << i);
        _k /= 2;
        ++i;
    }

    // create graph
    vertex *v_top = nullptr, *v_bot = nullptr;

    for(i = 0; i < l.layers.size(); ++i){
        vertex *_v_top = nullptr, *_v_bot = nullptr;

        if(i == 0) v_bot = l.v_source;
        if(i == l.layers.size() - 1) _v_bot = l.v_sink;
        else {
            _v_bot = init_vertex(l);
            _v_top = init_vertex(l);
        }

        init_edge(l, v_bot, _v_bot);
        init_edge(l, v_bot, _v_top);
        init_edge(l, v_top, _v_bot);
        init_edge(l, v_top, _v_top);

        if(i > 0 && l.layers[i - 1]){
            edge *_e = init_edge(l, v_bot, l.v_sink);
            _e->skips = l.layers.size() - i;
        }

        v_bot = _v_bot;
        v_top = _v_top;
    }

    l.V.push_back(l.v_sink);
}


// paths helpers & utils
//----------------------------------------------------------------------------------------------------------------------

inline path* get_path(ltls_single& l, uint32_t label){
    return l.P[label - 1];
}

void update_path(ltls_single& l, path *p){
    //D_COUT << "UPDATE PATH " << p->label << endl;
    p->score = 0;
    for(auto e : p->E){
        p->score = log(exp(e->score) + exp(p->score));
        if(SKIPS_FOR) for(int i = 0; i < e->skips - 1; ++i) p->score = log(exp(e->score) + exp(p->score));
    }
}

path* path_from_edges(ltls_single& l, vector<edge*> &E){
    uint32_t _label = 0, _layer = 0;
    vertex *_v = l.v_source;

    if(CHECKS) {
        if (E.back()->v_out != l.v_source) {
            cout << "This path doesn't start at graph source!\n";
            return nullptr;
        }

        if (E.front()->v_in != l.v_sink) {
            cout << "This path doesn't end at graph source!\n";
            return nullptr;
        }
    }

    for(auto e = E.rbegin(); e != E.rend(); ++e){
        for(int j = 0; j < MAX_E_OUT; ++j){
            if(_v->E[j] == (*e)){
                if(j == 1 && _v->E[j]->v_in == l.v_sink) _label += l.layers.back();
                else if(j == 1) _label += j << _layer;
                else if(j == 2) for(int k = _layer; k < l.layers.size(); ++k) _label += l.layers[k];
                break;
            }
        }
        ++_layer;
        _v = (*e)->v_in;
    }

    path *_p = l.P[_label];
    update_path(l, _p);
    return _p;
}

vector<edge*> get_top_path_from(ltls_single& l, vertex *v_source){
    D_COUT << "TOP PATH FROM: " << v_source->v << endl;

    for(auto v : l.V) v->score = numeric_limits<float>::lowest();
    v_source->score = 0;
    v_source->e_prev = nullptr;

    priority_queue<vertex*, vector<vertex*>, single_compare_vertex_ptr_functor> v_queue;
    v_queue.push(v_source);

    while(!v_queue.empty()){
        vertex *_v = v_queue.top();
        v_queue.pop();

        for(int i = 0; i < MAX_E_OUT; ++i){
            edge *_e = _v->E[i];
            if(_e && !_e->removed){
                float new_length = log(exp(_e->score) + exp(_v->score));
                if(SKIPS_FOR) for(int i = 0; i < _e->skips - 1; ++i) new_length = log(exp(_e->score) + exp(new_length));
                if(_e->v_in->score < new_length){
                    _e->v_in->score = new_length;
                    _e->v_in->e_prev = _e;
                    v_queue.push(_e->v_in);
                }
            }
        }
    }

    vector<edge*> top_path;
    vertex *_v = l.v_sink;
    while(_v != v_source){
        edge *_e = _v->e_prev;
        top_path.push_back(_e);
        _v = _e->v_out;
    }
    return top_path;
}

#define BRUTE_TOP_PATHS false
path* get_next_top(ltls_single& l) {

    if (BRUTE_TOP_PATHS) { // brute - update all paths' scores and sort
        if(l.P_ranking.size() == 0){
            for (auto p : l.P) l.P_ranking.push_back(p);
        }
        if (l.ranking_size == 0) {
            for (auto p : l.P) update_path(l, p);
            sort(l.P_ranking.rbegin(), l.P_ranking.rend(), single_compare_path_ptr_func);
        }
        return l.P_ranking[l.ranking_size++];

    } else {
        if (l.ranking_size == 0) { // first longest path
            vector<edge*> top_path = get_top_path_from(l, l.v_source);
            l.P_ranking.clear();
            l.P_potential_ranking = priority_queue<path*, vector<path*>, single_compare_path_ptr_functor>();
            l.P_ranking.push_back(path_from_edges(l, top_path));
        }
        else { // modified yen's algorithm
            list<edge*> root_path;
            uint32_t max_spur_edge = min(l.P_ranking.back()->E.size(), l.layers.size() - l.ranking_size); // lawler based modification
            for (int i = 0; i < max_spur_edge; ++i) {
                edge *spur_edge = l.P_ranking.back()->E[i];
                vertex *spur_vertex = spur_edge->v_out;

                for (auto p : l.P_ranking) {
                    if (root_path.size() <= p->E.size() && equal(root_path.rbegin(), root_path.rend(), p->E.begin()))
                        p->E[i]->removed = true;
                }

                bool all_removed = true;
                for(int j = 0; j < MAX_E_OUT; ++j){
                    if(spur_vertex->E[j] && !spur_vertex->E[j]->removed) all_removed = false;
                }

                if(!all_removed) {
                    vector<edge*> spur_path = get_top_path_from(l, spur_vertex);
                    vector<edge*> total_path;
                    total_path.insert(total_path.end(), spur_path.begin(), spur_path.end());
                    total_path.insert(total_path.end(), root_path.begin(), root_path.end());

                    l.P_potential_ranking.push(path_from_edges(l, total_path));
                }

                root_path.push_front(spur_edge);
                for(auto e : l.E) e->removed = false;
            }

            l.P_ranking.push_back(l.P_potential_ranking.top());
            l.P_potential_ranking.pop();
        }

        l.ranking_size = l.P_ranking.size();
        return l.P_ranking.back();
    }
}

vector<path*> top_k_paths(ltls_single& l, uint32_t top_k){
    D_COUT << "TOP " << top_k << " PATHS\n";

    vector<path*> top_paths;
    for(int k = 1; k <= top_k; ++k){
        top_paths.push_back(get_next_top(l));
    }

    return top_paths;
}

void check_graph(ltls_single& l){

    cout << "Checking edges..." << endl;
    int l_sink_in = 0, e_sink_in = 0;

    for(auto layer : l.layers){
        if(layer) ++l_sink_in;
    }

    for(auto e : l.E){
        if(!e->v_out) cout << "Edge " << e->e << " doesn't have v_out!\n";
        if(!e->v_in) cout << "Edge " << e->e << " doesn't have v_in!\n";
        else if(e->v_in == l.v_sink) ++e_sink_in;
    }

    if(e_sink_in == l_sink_in) cout << "Number of edges going to sink mismatch expected number " << e_sink_in << " != " << l_sink_in << endl;

    cout << "Checking paths..." << endl;
    if(l.k != l.P.size()) cout << "Number of paths mismatch number of labels " << l.k << " != " << l.P.size() << endl;
    for(auto p1 : l.P){
        if(p1->E.back()->v_in != l.v_sink) cout << "Path " << p1->label << " aren't ending at sink!\n";
        for(auto p2 : l.P){
            if(p1 != p2 && p1->E == p2->E){
                cout << "Path " << p1->label << " and " << p2->label << " are exactly the same paths!\n";
                cout << "Path " << p1->label << ":";
                for(auto e : p1->E) cout << " " << e->e;
                cout << "\nPath " << p2->label << ":";
                for(auto e : p2->E) cout << " " << e->e;
                cout << endl;
            }
        }
    }
}

// learn
//----------------------------------------------------------------------------------------------------------------------

void evaluate(ltls_single& l, base_learner& base, example& ec) {
    D_COUT << "EVALUATE ALL THE EDGES\n";

    l.ranking_size = 0;
    ec.l.simple = {FLT_MAX, 0.f, 0.f};
    for(auto e : l.E){
        base.predict(ec, e->e);
        e->score = ec.partial_prediction;
        if(SKIPS_MUL) e->score *= e->skips;
    }
}

//void compute_gradients(ltls_single& l, unordered_set<edge*> &positive_edges){
void compute_gradients(ltls_single& l, COST_SENSITIVE::label &ec_labels){

    for(auto v : l.V) v->alpha = v->beta = numeric_limits<float>::lowest();
    for(auto e : l.E) e->positive = 0;
    l.v_source->alpha = l.v_sink->beta = 0;

    for(auto e : l.E){
        float path_length = e->v_out->alpha + e->score;
        e->v_in->alpha = log(exp(e->v_in->alpha) + exp(path_length));
    }

    for(auto e = l.E.rbegin(); e != l.E.rend(); ++e){
        float path_length = (*e)->v_in->beta + (*e)->score;
        (*e)->v_out->beta = log(exp((*e)->v_out->beta) + exp(path_length));
    }

    if(MULTILABEL) {
        vector<path*> positive_paths, top_paths = top_k_paths(l, ec_labels.costs.size());

        for (auto& cl : ec_labels.costs) {
            path *_p = get_path(l, cl.class_index);
            positive_paths.push_back(_p);
            update_path(l, _p);
        }

        for (auto &tp : top_paths) {
            bool negative_path = true;
            for (auto &pp : positive_paths) {
                if (tp == pp) {
                    negative_path = false;
                    break;
                }
            }
            if (negative_path) {
                for (auto e : tp->E) {
                    --e->positive;
                }
            }
        }
    } else {
        for (auto& cl : ec_labels.costs) {
            path *_p = get_path(l, cl.class_index);
            for(auto e : _p->E) ++e->positive;
        }
    }

    for(auto e : l.E){
        float yuv = 0;
        float puvx = exp(e->v_out->alpha + e->score + e->v_in->beta - l.v_sink->alpha);
        if(e->positive > 0) yuv = 1;
        e->gradient = yuv - puvx;
    }

    if(DEBUG) {
        D_COUT << "\n\nSCORES: " << endl;
        for (auto e : l.E) {
            D_COUT << e->score << " ";
        }

        D_COUT << "\n\nALPHA: " << endl;
        for (auto v : l.V) {
            D_COUT << v->alpha << " ";
        }

        D_COUT << "\n\nBETA: " << endl;
        for (auto v : l.V) {
            D_COUT << v->beta << " ";
        }

        D_COUT << "\n\nGRADIENTS: " << endl;
        for (auto e : l.E) {
            D_COUT << e->gradient << " ";
        }

        D_COUT << "\n\n";
    }
}

void update_edges(ltls_single& l, base_learner& base, example& ec){

    ec.l.simple = {1.f, 1.f, 0.f};
    for(auto e : l.E){
        D_COUT << "LEARN EDGE: " << e->e << " GRADIENT: " << e->gradient << endl;
        l.loss_function->gradient = e->gradient;
        ec.partial_prediction = e->score;
        base.update(ec, e->e);
    }
}

void learn(ltls_single& l, base_learner& base, example& ec){
    D_COUT << "LEARN EXAMPLE\n";

    COST_SENSITIVE::label ec_labels = ec.l.cs; // copy is needed to restore example later
    if (!ec_labels.costs.size()) return;

    // evaluate
    evaluate(l, base, ec);
    ec.loss = l.loss_function->loss = 1;

    // check loss
    //vector<path*> positive_paths;
//    unordered_set<edge*> positive_edges;
//
//    for (auto& cl : ec_labels.costs) {
//        path *_p = get_path(l, cl.class_index);
//        //positive_paths.push_back(_p);
//        for(auto e : _p->E) positive_edges.insert(e);
//    }

//    vector<path*> top_paths = top_k_paths(l, ec_labels.costs.size());
//    sort(positive_paths.rbegin(), positive_paths.rend(), single_compare_path_ptr_func);
//    path *positive_path = nullptr, *negative_path = nullptr;
//
//    for(auto& tp : top_paths){
//        for(auto& pp : positive_paths){
//            negative_path = tp;
//            if(tp == pp){
//                negative_path = nullptr;
//                break;
//            }
//        }
//        if(negative_path){
//            positive_path = positive_paths.back();
//            ec.loss = l.loss_function->loss = 1 + negative_path->score - positive_path->score;
//            break;
//        }
//    }

    //if(ec.loss) {
    compute_gradients(l, ec_labels);
    update_edges(l, base, ec);
    //}

    ec.l.cs = ec_labels;
    //ec.pred.multiclass = positive_paths.front()->label;

    int x;
    if(DEBUG) cin >> x;

    D_COUT << "----------------------------------------------------------------------------------------------------\n";

}

// predict
//----------------------------------------------------------------------------------------------------------------------

void predict(ltls_single& l, base_learner& base, example& ec){
    D_COUT << "PREDICT EXAMPLE\n";

    ++l.prediction_count;

    COST_SENSITIVE::label ec_labels = ec.l.cs;

    evaluate(l, base, ec);
    vector<path*> top_paths = top_k_paths(l, l.p_at_k);

    vector<uint32_t> true_labels;
    for (auto &cl : ec_labels.costs) true_labels.push_back(cl.class_index);

    if (l.p_at_k > 0 && true_labels.size() > 0) {
        for (size_t i = 0; i < l.p_at_k; ++i) {
            if (find(true_labels.begin(), true_labels.end(), top_paths[i]->label) != true_labels.end())
                l.precision_at_k[i] += 1.0f;
        }
    }

    ec.l.cs = ec_labels;
    ec.pred.multiclass = top_paths.front()->label;
}


// finish
//----------------------------------------------------------------------------------------------------------------------

void finish_example(vw& all, ltls_single& l, example& ec) {
    VW::finish_example(all, &ec);
}

void end_pass(ltls_single& l){
    cout << "end of pass " << l.all->passes_complete << "\n";
}

void finish(ltls_single& l){
    float correct = 0;
    for (size_t i = 0; i < l.p_at_k; ++i) {
        correct += l.precision_at_k[i];
        cout << "P@" << i + 1 << " = " << correct / (l.prediction_count * (i + 1)) << "\n";
    }
}

// setup
//----------------------------------------------------------------------------------------------------------------------

base_learner* ltls_single_setup(vw& all) //learner setup
{
    if (missing_option<size_t, true>(all, "ltls_single", "Log time log space for multilabel with <k> labels"))
        return nullptr;
    new_options(all, "ltls_single options")
            ("p_at", po::value<uint32_t>(), "P@k (default 1)");
    add_options(all);

    ltls_single& data = calloc_or_throw<ltls_single>();
    data.k = all.vm["ltls_single"].as<size_t>();
    data.all = &all;

    data.precision = 0;
    data.predicted_number = 0;
    data.prediction_count = 0;
    data.p_at_k = 1;

    // ltls parse options
    //------------------------------------------------------------------------------------------------------------------

    learner<ltls_single> *l;

    if( all.vm.count("p_at") ) data.p_at_k = all.vm["p_at"].as<uint32_t>();
    data.precision_at_k.resize(data.p_at_k);

    // ltls init graph and paths
    //------------------------------------------------------------------------------------------------------------------

    init_graph(data);
    init_paths(data);

    data.loss_function = new ltls_loss_function();
    all.loss = data.loss_function;

    if(CHECKS) check_graph(data);

    l = &init_multiclass_learner(&data, setup_base(all), learn, predict, all.p, data.e);

    // override parser
    //------------------------------------------------------------------------------------------------------------------

    all.p->lp = COST_SENSITIVE::cs_label;
    all.cost_sensitive = make_base(*l);

    all.holdout_set_off = true; // turn off stop based on holdout loss

    // log info & add some event handlers
    //------------------------------------------------------------------------------------------------------------------
    cout << "ltls_single\nk = " << data.k << "\nv = " << data.v
         << "\ne = " << data.e << "\nlayers = " << data.layers.size() << endl;

//    if(path_assignment_policy.length())
//        cout << path_assignment_policy << endl;

    //l->set_finish_example(finish_example);
    l->set_end_pass(end_pass);
    l->set_finish(finish);

    return all.cost_sensitive;
}
