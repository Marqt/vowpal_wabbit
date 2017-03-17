#include <float.h>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <queue>
#include <list>
#include <limits>
#include <random>
#include <chrono>

#include "reductions.h"
#include "vw.h"

using namespace std;
using namespace LEARNER;

#define SKIPS_MUL true
#define STATS true
#define EXP exp
#define LOG log

#define DEBUG false
#define D_COUT if(DEBUG) cout

// experimental fast math
//----------------------------------------------------------------------------------------------------------------------

static inline float
fasterpow2 (float p)
{
    float clipp = (p < -126) ? -126.0f : p;
    union { uint32_t i; float f; } v = { static_cast<uint32_t> ( (1 << 23) * (clipp + 126.94269504f) ) };
    return v.f;
}

static inline float
fasterexp (float p)
{
    return fasterpow2 (1.442695040f * p);
}

static inline float
fasterlog2 (float x)
{
    union { float f; uint32_t i; } vx = { x };
    float y = vx.i;
    y *= 1.1920928955078125e-7f;
    return y - 126.94269504f;
}

static inline float
fasterlog (float x)
{
    union { float f; uint32_t i; } vx = { x };
    float y = vx.i;
    y *= 8.2629582881927490e-8f;
    return y - 87.989971088f;
}

// custom loss class
//----------------------------------------------------------------------------------------------------------------------

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


// graph's structs
//----------------------------------------------------------------------------------------------------------------------

struct edge;
struct vertex;

struct edge{
    uint32_t e;

    vertex *v_out;
    vertex *v_in;
    uint32_t skips;

    // for longest path and yen
    bool removed;

    // for gradients
    float gradient;
};

#define MAX_E_OUT 3
struct vertex{
    uint32_t v;
    size_t e_out;
    edge* E[MAX_E_OUT];

    // for longest path and yen
    float score;
    edge *e_prev;

    // for gradients
    float alpha;
    float beta;

    bool operator < (const vertex& r) const { return score < r.score; }
};

struct path{
    uint32_t label;
    vector<edge*> E;
    float score;

    bool operator < (const path& r) const { return score < r.score; }
};

struct model_path{
    uint32_t label;
    float score;
    path *p;

    bool operator < (const model_path& r) const { return score < r.score; }
};


// struct helpers
//----------------------------------------------------------------------------------------------------------------------

bool compare_model_path_ptr_func(const model_path* l, const model_path* r) { return (*l < *r); }
struct compare_model_path_ptr_functor{ bool operator()(const model_path* l, const model_path* r) const { return (*l < *r); } };

bool compare_path_ptr_func(const path* l, const path* r) { return (*l < *r); }
struct compare_path_ptr_functor{ bool operator()(const path* l, const path* r) const { return (*l < *r); } };

bool compare_vertex_ptr_func(const vertex* l, const vertex* r) { return (*l < *r); }
struct compare_vertex_ptr_functor{ bool operator()(const vertex* l, const vertex* r) const { return (*l < *r); } };


// models
//----------------------------------------------------------------------------------------------------------------------

struct model {
    uint32_t base;

    // model's paths and ranking
    uint32_t ranking_size;
    vector<model_path*> P_ranking;
    priority_queue<model_path*, vector<model_path*>, compare_model_path_ptr_functor> P_potential_ranking;
    model_path *P;
    model_path **L;

    // model's predictions
    polyprediction* e_pred;
};

struct ltls {
    vw* all;

    // graph
    uint32_t k; // number of labels/paths
    uint32_t e; // number of edges
    uint32_t v; // number of vertices
    vector<uint32_t> layers;

    vertex *v_source;
    vertex *v_sink;

    vertex *_V;
    vector<vertex*> V; // vertices in topological order
    edge *_E;
    vector<edge*> E; // edges in topological order
    path *_P;
    vector<path*> P; // paths

    // models
    uint32_t ensemble;
    uint32_t models_to_use;
    model *M;

    bool random_policy;
    ltls_loss_function *loss_function;

    uint32_t p_at_k;
    uint32_t max_rank;

    vector<uint32_t>(*top_k_ensemble)(ltls& l, uint32_t top_k);

    // stats
    float precision;
    float predicted_number;
    vector<float> precision_at_k;
    size_t learn_count;
    size_t predict_count;

    uint32_t get_next_top_count;
    uint32_t get_path_score_count;
    uint32_t sum_rank;

    chrono::time_point<chrono::steady_clock> learn_predict_start_time_point;
};


// inits
//----------------------------------------------------------------------------------------------------------------------

void init_models(ltls& l){
    D_COUT << "INIT MODELS\n";

    default_random_engine rng;
    rng.seed(l.all->random_seed);

    l.M = calloc_or_throw<model>(l.ensemble);
    for(uint32_t i = 0; i < l.ensemble; ++i){
        model& m = l.M[i];
        m.base = i * l.k;

        m.e_pred = calloc_or_throw<polyprediction>(l.e);
        m.P = calloc_or_throw<model_path>(l.k);
        m.L = calloc_or_throw<model_path*>(l.k);

        for(uint32_t p = 0; p < l.k; ++p){
            m.P[p].p = l.P[p];
        }

        if(l.random_policy){
            for(uint32_t p = 0; p < l.k - 1; ++p){
                uniform_int_distribution <uint32_t> dist(p, l.k - 1);
                size_t swap = dist(rng);
                auto temp = m.P[p];
                m.P[p] = m.P[swap];
                m.P[swap] = temp;
            }
        }

        for(uint32_t p = 0; p < l.k; ++p){
            m.P[p].label = p + 1;
            m.L[m.P[p].p->label - 1] = &m.P[p];
        }
    }
}

edge* init_edge(ltls& l, vertex *v_out, vertex *v_in){
    if(!v_in || !v_out || v_in == v_out) return nullptr;

    //edge *e = new edge();
    edge *e = &l._E[l.e];
    e->e = l.e++;
    e->v_out = v_out;
    e->v_in = v_in;
    e->skips = 1;
    e->removed = false;

    for(size_t i = 0; i < MAX_E_OUT; ++i){
        if(!e->v_out->E[i]){
            e->v_out->E[i] = e;
            e->v_out->e_out = i + 1;
            break;
        }
    }

    l.E.push_back(e);
    return e;
}

vertex* init_vertex(ltls& l){
    //vertex *v = new vertex();
    vertex *v = &l._V[l.v];
    v->v = l.v++;
    v->E[0] = nullptr;
    v->E[1] = nullptr;
    v->E[2] = nullptr;
    l.V.push_back(v);
    return v;
}

void init_paths(ltls& l){
    D_COUT << "INIT PATHS\n";

    l._P = calloc_or_throw<path>(l.k);
    l.P.reserve(l.k);

    for(uint32_t label = 1; label <= l.k; ++label) {
        //path *p = new path();
        path *p = &l._P[label - 1];
        p->score = 0;
        p->label = label;
        uint32_t _label, _layers;

        _label = label - 1;
        _layers = l.layers.size();
        while (_label >= l.layers[_layers - 1]) {
            _label -= l.layers[_layers - 1];
            --_layers;
        }

        vertex *_v = l.v_source;
        edge *_e;

        for(size_t i = 0; i < _layers; ++i) {
            _e = _v->E[_label % 2];
            _label = _label >> 1;
            p->E.push_back(_e);
            _v = _e->v_in;
        }

        if (_v != l.v_sink) {
            for(size_t i = _v->e_out - 1; i >= 0; --i) {
                if (_v->E[i]->v_in == l.v_sink) {
                    p->E.push_back(_v->E[i]);
                    break;
                }
            }
        }

        l.P.push_back(p);
    }
}

void init_graph(ltls& l){
    D_COUT << "INIT GRAPH\n";

    // calc layers, vertices, edges;
    size_t i = 0;
    uint32_t _k = l.k;
    while(_k > 0){
        l.layers.push_back((_k % 2) << i);
        _k /= 2;
        ++i;
    }

    size_t _v = l.layers.size() * 2;
    l._V = calloc_or_throw<vertex>(_v);
    l.V.reserve(_v);

    size_t _e = (l.layers.size() - 1) * 4;
    for(auto _l : l.layers) if(_l) ++_e;
    l._E = calloc_or_throw<edge>(_e);
    l.E.reserve(_e);

    // create graph
    l.e = l.v = 0;
    l.v_source = init_vertex(l);
    l.v_sink = init_vertex(l);
    l.V.pop_back();

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

inline path* get_path(ltls& l, model& m, uint32_t label){
    return m.P[label - 1].p;
}

void update_path(model& m, path *p){
    p->score = 0;
    for(auto e : p->E) {
        //p->score = LOG(EXP(m.e_pred[e->e].scalar) + EXP(p->score));
        p->score += m.e_pred[e->e].scalar;
    }
    m.L[p->label - 1]->score = p->score;
}

float get_label_score(ltls& l, model& m, uint32_t label){
    if(STATS) ++l.get_path_score_count;
    update_path(m, m.P[label - 1].p);
    return m.P[label - 1].score;
}

model_path* path_from_edges(ltls& l, model& m, vector<edge*> &E){
    uint32_t _label = 0, _layer = 0;
    vertex *_v = l.v_source;

    for(auto e = E.rbegin(); e != E.rend(); ++e){
        for(int j = 0; j < _v->e_out; ++j){
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

    path *_p = m.L[_label]->p;
    update_path(m, _p);

    return m.L[_label];
}

template<bool graph_source>
vector<edge*> get_top_path_from(ltls& l, model& m, vertex *v_source){
    D_COUT << "TOP PATH FROM: " << v_source->v << endl;

    for(auto v : l.V) v->score = numeric_limits<float>::lowest();
    v_source->score = 0;
    v_source->e_prev = nullptr;

    if(graph_source){
        for(auto _v : l.V){
            for(uint32_t i = 0; i < _v->e_out; ++i){
                edge *_e = _v->E[i];
                //float new_length = LOG(EXP(m.e_pred[_e->e].scalar) + EXP(_v->score));
                //float new_length = m.e_pred[_e->e].scalar * l.E[i]->skips + _v->score;
                float new_length = m.e_pred[_e->e].scalar + _v->score;
                if(_e->v_in->score < new_length){
                    _e->v_in->score = new_length;
                    _e->v_in->e_prev = _e;
                }
            }
        }
    }
    else {
        unordered_set <vertex *> v_set;
        queue <vertex *> v_queue;
        v_queue.push(v_source);

        while (!v_queue.empty()) {
            vertex *_v = v_queue.front();
            v_queue.pop();

            for (uint32_t i = 0; i < _v->e_out; ++i) {
                edge *_e = _v->E[i];
                if (!_e->removed) {
                    //float new_length = LOG(EXP(m.e_pred[_e->e].scalar) + EXP(_v->score));
                    float new_length = m.e_pred[_e->e].scalar + _v->score;
                    if (_e->v_in->score <= new_length) {
                        _e->v_in->score = new_length;
                        _e->v_in->e_prev = _e;
                    }
                    if (v_set.find(_e->v_in) == v_set.end()) {
                        v_set.insert(_e->v_in);
                        v_queue.push(_e->v_in);
                    }
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

model_path* get_next_top(ltls& l, model& m) {

    if(STATS) ++l.get_next_top_count;

    if (m.ranking_size == 0) { // first longest path
        vector<edge*> top_path = get_top_path_from<true>(l, m, l.v_source);
        m.P_ranking.clear();
        m.P_potential_ranking = priority_queue<model_path*, vector<model_path*>, compare_model_path_ptr_functor>();
        m.P_ranking.push_back(path_from_edges(l, m, top_path));
    }
    else { // modified yen's algorithm
        list<edge*> root_path;
        uint32_t max_spur_edge = min(m.P_ranking.back()->p->E.size(), l.layers.size() - m.ranking_size); // lawler based modification
        for(uint32_t i = 0; i < max_spur_edge; ++i) {
            edge *spur_edge = m.P_ranking.back()->p->E[i];
            vertex *spur_vertex = spur_edge->v_out;

            for(auto p : m.P_ranking) {
                if (root_path.size() <= p->p->E.size() && equal(root_path.rbegin(), root_path.rend(), p->p->E.begin()))
                    p->p->E[i]->removed = true;
            }

            bool all_removed = true;
            for(uint32_t j = 0; j < spur_vertex->e_out; ++j){
                if(!spur_vertex->E[j]->removed) all_removed = false;
            }

            if(!all_removed) {
                vector<edge*> spur_path = get_top_path_from<false>(l, m, spur_vertex);
                vector<edge*> total_path;
                total_path.insert(total_path.end(), spur_path.begin(), spur_path.end());
                total_path.insert(total_path.end(), root_path.begin(), root_path.end());

                m.P_potential_ranking.push(path_from_edges(l, m, total_path));
            }

            root_path.push_front(spur_edge);
            for(auto e : l.E) e->removed = false;
        }

        m.P_ranking.push_back(m.P_potential_ranking.top());
        m.P_potential_ranking.pop();
    }

    m.ranking_size = m.P_ranking.size();
    return m.P_ranking.back();
}

vector<model_path*> top_k_paths(ltls& l, model& m, uint32_t top_k){
    D_COUT << "TOP " << top_k << " PATHS\n";

    vector<model_path*> top_paths;
    for(uint32_t k = 0; k < top_k; ++k){
        top_paths.push_back(get_next_top(l, m));
    }

    return top_paths;
}


// ensemble top
//----------------------------------------------------------------------------------------------------------------------

template<bool single_ltls>
vector<uint32_t> top_k_ensemble(ltls& l, uint32_t top_k){
    D_COUT << "TOP ENSEMBLE" << top_k << " PATHS\n";

    if(single_ltls){
        vector<model_path*> top_paths = top_k_paths(l, l.M[0], top_k);
        vector<uint32_t> top_labels;
        for(auto p : top_paths) top_labels.push_back(p->label);
        return top_labels;
    }
    else { // threshold algorithm
        int rank = 0;
        unordered_set <uint32_t> seen_labels;
        vector<pair<float, uint32_t>> _top_labels;
        float threshold;

        do {
            if(rank >= l.max_rank) break;
            //if(rank == l.k) break; // not really needed
            ++rank;

            vector<model_path*> paths_at_rank;
            unordered_map<uint32_t, pair<uint32_t, float>> random_except;
            unordered_set<uint32_t> new_labels;

            for(uint32_t i = 0; i < l.models_to_use; ++i) {
                model_path *_p = get_next_top(l, l.M[i]);
                paths_at_rank.push_back(_p);
                random_except.insert({_p->label, {i, _p->score}});

                if (seen_labels.find(_p->label) == seen_labels.end()) new_labels.insert(_p->label);
                seen_labels.insert(_p->label);
            }

            for(auto label : new_labels) {
                float label_aggregate_score = 0;
                for(uint32_t i = 0; i < l.models_to_use; ++i) {
                    if (i == random_except[label].first) label_aggregate_score += random_except[label].second;
                    else label_aggregate_score += get_label_score(l, l.M[i], label);
                }
                _top_labels.push_back({label_aggregate_score, label});
            }

            sort(_top_labels.rbegin(), _top_labels.rend());
            while (_top_labels.size() > top_k) _top_labels.pop_back();

            threshold = 0;
            for(auto p : paths_at_rank) threshold += p->score;

        } while (_top_labels.size() != top_k || _top_labels.back().first < threshold);

        vector<uint32_t> top_labels;
        for(auto p : _top_labels) top_labels.push_back(p.second);

        l.sum_rank += rank;

        return top_labels;
    }
}

// learn
//----------------------------------------------------------------------------------------------------------------------

inline void evaluate(ltls& l, model& m, base_learner& base, example& ec) {
    D_COUT << "EVALUATE ALL THE EDGES\n";

    m.ranking_size = 0;
    ec.l.simple = {FLT_MAX, 0.f, 0.f};
    base.multipredict(ec, m.base, l.e, m.e_pred, false);
    if(SKIPS_MUL) for(size_t i = 0; i < l.e; ++i) m.e_pred[i].scalar *= l.E[i]->skips;
}

void compute_gradients(ltls& l, model& m, COST_SENSITIVE::label &ec_labels){

    for(auto v : l.V) v->alpha = v->beta = numeric_limits<float>::lowest();
    l.v_source->alpha = l.v_sink->beta = 0;

    for(auto e : l.E){
        float path_length = e->v_out->alpha + m.e_pred[e->e].scalar;
        e->v_in->alpha = LOG(EXP(e->v_in->alpha) + EXP(path_length));
    }

    for(auto e = l.E.rbegin(); e != l.E.rend(); ++e){
        float path_length = (*e)->v_in->beta + m.e_pred[(*e)->e].scalar;
        (*e)->v_out->beta = LOG(EXP((*e)->v_out->beta) + EXP(path_length));
    }

    unordered_set<edge*> positive_edges;
    for(auto& cl : ec_labels.costs) {
        path *_p = get_path(l, m, cl.class_index);
        for(auto e : _p->E) positive_edges.insert(e);
    }

    for(auto e : l.E){
        float yuv = 0;
        if(positive_edges.find(e) != positive_edges.end()) yuv = 1;
        float puvx = EXP(e->v_out->alpha + m.e_pred[e->e].scalar + e->v_in->beta - l.v_sink->alpha);
        e->gradient = yuv - puvx;
    }
}

void update_edges(ltls& l, model& m, base_learner& base, example& ec){
    ec.l.simple = {1.f, 0.f, 0.f};
    for(auto e : l.E){
        l.loss_function->gradient = e->gradient;
        ec.partial_prediction = m.e_pred[e->e].scalar;
        base.update(ec, m.base + e->e);
    }
}

void learn(ltls& l, base_learner& base, example& ec){
    D_COUT << "LEARN EXAMPLE\n";

    if(STATS && !l.learn_count) l.learn_predict_start_time_point = chrono::steady_clock::now();

    COST_SENSITIVE::label ec_labels = ec.l.cs; // copy is needed to restore example later
    if (!ec_labels.costs.size()) return;

    for(uint32_t i = 0; i < l.ensemble; ++i) {
        model& m = l.M[i];

        evaluate(l, m, base, ec);
        ec.loss = l.loss_function->loss = 1;

        compute_gradients(l, m, ec_labels);
        update_edges(l, m, base, ec);
    }

    ec.l.cs = ec_labels;
    ec.pred.multiclass = 0;

    D_COUT << "----------------------------------------------------------------------------------------------------\n";
}

// predict
//----------------------------------------------------------------------------------------------------------------------

void predict(ltls& l, base_learner& base, example& ec){
    D_COUT << "PREDICT EXAMPLE\n";

    if(STATS && !l.predict_count) l.learn_predict_start_time_point = chrono::steady_clock::now();

    COST_SENSITIVE::label ec_labels = ec.l.cs;

    for(uint32_t i = 0; i < l.models_to_use; ++i) evaluate(l, l.M[i], base, ec);
    auto top_paths = l.top_k_ensemble(l, l.p_at_k);

    if(STATS) {
        vector <uint32_t> true_labels;
        for (auto &cl : ec_labels.costs) true_labels.push_back(cl.class_index);

        if (l.p_at_k > 0 && true_labels.size() > 0) {
            for (size_t i = 0; i < l.p_at_k; ++i) {
                if (find(true_labels.begin(), true_labels.end(), top_paths[i]) != true_labels.end())
                    l.precision_at_k[i] += 1.0f;
            }
        }
    }

    ++l.predict_count;
    ec.l.cs = ec_labels;
    ec.pred.multiclass = top_paths.front();
}


// finish
//----------------------------------------------------------------------------------------------------------------------

void finish_example(vw& all, ltls& l, example& ec) {
    all.sd->update(ec.test_only, 0, ec.weight, ec.num_features);
    VW::finish_example(all, &ec);
}

void end_pass(ltls& l){
    cout << "end of pass " << l.all->passes_complete << "\n";
}

void finish(ltls& l){
    if(STATS){
        float correct = 0;
        for(size_t i = 0; i < l.p_at_k; ++i) {
            correct += l.precision_at_k[i];
            cout << "P@" << i + 1 << " = " << correct / (l.predict_count * (i + 1)) << endl;
        }

        cout << "get_next_top_count = " << l.get_next_top_count << "\nget_path_score_count = " << l.get_path_score_count << endl;
        float average_rank = static_cast<float>(l.sum_rank)/l.predict_count;
        cout << "average_rank = " << average_rank << endl;

        auto end_time_point = chrono::steady_clock::now();
        auto learn_predict_time = end_time_point - l.learn_predict_start_time_point;
        cout << "learn_predict_time = " << static_cast<double>(chrono::duration_cast<chrono::microseconds>(learn_predict_time).count()) / 1000000 << "s\n";
    }

//    for(size_t i = 0; i < l.ensemble; ++i){
//        free(l.M->e_pred);
//        free(l.M->P);
//        free(l.M->L);
//    }
//
//    free(l.M);
//    free(l._V);
//    free(l._E);
//    free(l._P);
}

void save_load_graph(ltls& l, io_buf& model_file, bool read, bool text){
    D_COUT << "SAVE/LOAD\n";

    if (model_file.files.size() > 0) {
        bool resume = l.all->save_resume;
        stringstream msg;
        msg << ":" << resume << "\n";
        bin_text_read_write_fixed(model_file, (char*) &resume, sizeof(resume), "", read, msg, text);

        // read/write P lookup
        for(uint32_t i = 0; i < l.ensemble; ++i) {
            model &m = l.M[i];
            for(uint32_t j = 0; j < l.k; ++j){
                uint32_t label_for_j = m.P[j].p->label;
                bin_text_read_write_fixed(model_file, (char *) &label_for_j, sizeof(label_for_j), "", read, msg, text);
                if(read) m.P[j].p = l.P[label_for_j - 1];
            }
            if(read) {
                for(uint32_t p = 0; p < l.k; ++p){
                    m.P[p].label = p + 1;
                    m.L[m.P[p].p->label - 1] = &m.P[p];
                }
            }
        }
    }
}

// setup
//----------------------------------------------------------------------------------------------------------------------

base_learner* ltls_setup(vw& all) //learner setup
{
    if (missing_option<size_t, true>(all, "ltls", "Log Time Log Space with log loss for multiclass"))
        return nullptr;
    new_options(all, "ltls options")
            ("ensemble", po::value<uint32_t>(), "number of ltls to ensemble (default 1)")
            ("models_to_use", po::value<uint32_t>(), "number of ltls to use in ensemble (default --ensemble value)")
            ("max_rank", po::value<uint32_t>(), "max rank in threshold algorithm (default --p_at value)")
            ("inorder_policy", "inorder path assignment policy")
            ("random_policy", "random path assignment policy")
            ("best_policy", "best possible path assignment policy")
            ("mixed_policy", po::value<float>(), "random mixed with best possible path policy")
            ("p_at", po::value<uint32_t>(), "P@k (default 1)");
    add_options(all);

    ltls& data = calloc_or_throw<ltls>();

    if(STATS){
        data.get_next_top_count = 0;
        data.get_path_score_count = 0;
        data.predict_count = 0;
        data.learn_count = 0;
    }

    data.k = all.vm["ltls"].as<size_t>();
    data.ensemble = 1;
    data.random_policy = false;
    data.all = &all;

    data.precision = 0;
    data.predicted_number = 0;
    data.predict_count = 0;
    data.p_at_k = 1;

    // ltls parse options
    //------------------------------------------------------------------------------------------------------------------

    learner<ltls> *l;
    string path_assignment_policy = "inorder_policy";

    if(all.vm.count("ensemble")) data.ensemble = all.vm["ensemble"].as<uint32_t>();
    *(all.file_options) << " --ensemble " << data.ensemble;
    data.models_to_use = data.ensemble;

    if(all.vm.count("models_to_use")) data.models_to_use = all.vm["models_to_use"].as<uint32_t>();
    if(data.models_to_use > data.ensemble) data.models_to_use = data.ensemble;
    if(data.models_to_use == 1) data.top_k_ensemble = top_k_ensemble<true>;
    else data.top_k_ensemble = top_k_ensemble<false>;

    if( all.vm.count("p_at") ) data.p_at_k = all.vm["p_at"].as<uint32_t>();
    data.precision_at_k.resize(data.p_at_k);
    data.max_rank = data.p_at_k;

    if( all.vm.count("max_rank") ) data.max_rank = all.vm["max_rank"].as<uint32_t>();

    if(all.vm.count("random_policy") || data.ensemble > 1) {
        path_assignment_policy = "random_policy";
        data.random_policy = true;
        //*(all.file_options) << " --random_policy";
    }

    // ltls init graph and paths
    //------------------------------------------------------------------------------------------------------------------

    init_graph(data);
    init_paths(data);
    init_models(data);

    data.loss_function = new ltls_loss_function();
    all.loss = data.loss_function;

    l = &init_multiclass_learner(&data, setup_base(all), learn, predict, all.p, data.e * data.ensemble);

    // override parser
    //------------------------------------------------------------------------------------------------------------------

    all.p->lp = COST_SENSITIVE::cs_label;
    all.cost_sensitive = make_base(*l);

    all.holdout_set_off = true; // turn off stop based on holdout loss

    // log info & add some event handlers
    //------------------------------------------------------------------------------------------------------------------

    cout << "ltls\nk = " << data.k << "\nv = " << data.v
         << "\ne = " << data.e << "\nlayers = " << data.layers.size() << endl;

    if(data.ensemble > 1) {
        cout << "ensemble = " << data.ensemble
             << "\nmodels_to_use = " << data.models_to_use
             << "\nmax_rank = " << data.max_rank << endl;
    }

    if(path_assignment_policy.length()) cout << path_assignment_policy << endl;

    l->set_save_load(save_load_graph);
    l->set_finish_example(finish_example);
    l->set_end_pass(end_pass);
    l->set_finish(finish);

    return all.cost_sensitive;
}
