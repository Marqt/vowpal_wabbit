#include <float.h>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <vector>
#include <queue>
#include <list>
#include <limits>
#include <random>
#include <chrono>
#include <fstream>

#include "reductions.h"
#include "vw.h"

using namespace std;
using namespace LEARNER;

#define REAL double // greater resistance to overflow/underflow on exp and log

static inline float safe_exp(REAL v){
    errno = 0;
    REAL _v = exp(v);
    if(isfinite(_v)) return _v;
    else return 0; // ?
}

static inline float safe_log(REAL v){
    errno = 0;
    REAL _v = exp(v);
    if(isfinite(_v)) return _v;
    else return 0; // ?
}

#define EXP exp
#define LOG log
#define SKIPS_MUL true


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
    REAL score;
    edge *e_prev;

    // for gradients
    REAL alpha;
    REAL beta;

    bool operator < (const vertex& r) const { return score < r.score; }
};

struct path{
    uint32_t label;
    vector<edge*> E;
    REAL score;

    bool operator < (const path& r) const { return score < r.score; }
};

struct model_path{
    uint32_t label;
    uint32_t ec;
    REAL score;
    path *p;

    bool operator < (const model_path& r) const { return score < r.score; }
};


// structs' helpers
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
    vector<model_path*> P_potential_ranking;
    model_path *P;
    model_path **L;

    // model's predictions
    polyprediction* e_pred;
    set<path*> available_paths;

    // eg
    float eg_P;
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

    bool random_policy, best_policy, mixed_policy, runtime_path_assignment;
    float mixed_p;
    ltls_loss_function *loss_function;

    uint32_t p_at_k;
    uint32_t max_rank;

    vector<uint32_t>(*top_k_ensemble)(ltls& l, uint32_t top_k);
    vector<model_path*>(*top_k_paths)(ltls& l, model& m, uint32_t top_k);
    vector<uint32_t> (*ta_top_k)(ltls& l, uint32_t top_k);

    // stats
    float precision;
    float predicted_number;
    vector<float> precision_at_k;
    size_t learn_count;
    size_t predict_count;
    bool stats; // more stats

    uint32_t get_next_top_count;
    uint32_t get_path_score_count;
    uint32_t sum_rank;

    chrono::time_point<chrono::steady_clock> learn_predict_start_time_point;
    uint32_t pass_count;

    // runtime path assignment
    default_random_engine rng;
    unordered_set<uint32_t> seen_labels;

    // eg
    bool learn_eg;
    bool predict_eg;
    float *eg_P;
    string eg_model;

    // l1_const
    bool l1_const_reg;
    float l1_const;

};


// inits
//----------------------------------------------------------------------------------------------------------------------

void init_models(ltls& l){

    l.M = calloc_or_throw<model>(l.ensemble);
    for(uint32_t i = 0; i < l.ensemble; ++i){
        model& m = l.M[i];
        m.base = i * l.k;

        m.e_pred = calloc_or_throw<polyprediction>(l.e);
        m.P = calloc_or_throw<model_path>(l.k);
        m.L = calloc_or_throw<model_path*>(l.k);

        m.eg_P = 1;

        for(uint32_t p = 0; p < l.k; ++p)
            m.P[p].p = l.P[p];

        if(l.random_policy){
            for(uint32_t p = 0; p < l.k - 1; ++p){
                uniform_int_distribution <uint32_t> dist(p, l.k - 1);
                size_t swap = dist(l.rng);
                auto temp = m.P[p];
                m.P[p] = m.P[swap];
                m.P[swap] = temp;
            }
        }
        else if(l.runtime_path_assignment){
            m.available_paths = set<path*>();
            for(auto p : l.P)
                m.available_paths.insert(p);
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

inline void update_path(ltls& l, model& m, path *p){
    p->score = 0;
    for(auto e : p->E) {
        //p->score = LOG(EXP(m.e_pred[e->e].scalar) + EXP(p->score));
        p->score += m.e_pred[e->e].scalar;
    }
    m.L[p->label - 1]->ec = l.predict_count;
    m.L[p->label - 1]->score = m.eg_P * p->score;
}

inline void update_path(ltls& l, model& m, model_path *p){
    update_path(l, m, p->p);
}

float get_label_score(ltls& l, model& m, uint32_t label){
    if(m.P[label - 1].ec != l.predict_count) {
        ++l.get_path_score_count;
        update_path(l, m, m.P[label - 1].p);
    }
    return m.P[label - 1].score;
}

model_path* path_from_edges(ltls& l, model& m, vector<edge*> &E){
    uint32_t _label = 0, _layer = 0;
    vertex *_v = l.v_source;
    float score = 0;

    for(auto e = E.rbegin(); e != E.rend(); ++e){
        for(size_t j = 0; j < _v->e_out; ++j){
            if(_v->E[j] == (*e)){
                if(j == 1 && _v->E[j]->v_in == l.v_sink) _label += l.layers.back();
                else if(j == 1) _label += j << _layer;
                else if(j == 2) for(size_t k = _layer; k < l.layers.size(); ++k) _label += l.layers[k];
                break;
            }
        }
        ++_layer;
        _v = (*e)->v_in;
        score += m.e_pred[(*e)->e].scalar;
    }

    m.L[_label]->ec = l.predict_count;
    m.L[_label]->score = m.eg_P * score;

    return m.L[_label];
}

template<bool graph_source>
vector<edge*> get_top_path_from(ltls& l, model& m, vertex *v_source){
    for(auto v : l.V) v->score = numeric_limits<float>::lowest();
    v_source->score = 0;
    v_source->e_prev = nullptr;

    if(graph_source){
        for(auto _v : l.V){
            for(uint32_t i = 0; i < _v->e_out; ++i){
                edge *_e = _v->E[i];
                //float new_length = LOG(EXP(m.e_pred[_e->e].scalar) + EXP(_v->score));
                float new_length = m.e_pred[_e->e].scalar + _v->score;
                if(_e->v_in->score < new_length){
                    _e->v_in->score = new_length;
                    _e->v_in->e_prev = _e;
                }
            }
        }
    }
    else {
        unordered_set <vertex*> v_set;
        queue <vertex*> v_queue;
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

//yen
model_path* get_next_top(ltls& l, model& m) {

    ++l.get_next_top_count;

    if (m.ranking_size == 0) { // first longest path
        vector<edge*> top_path = get_top_path_from<true>(l, m, l.v_source);
        m.P_ranking.clear();
        m.P_potential_ranking.clear();
        m.P_ranking.push_back(path_from_edges(l, m, top_path));
    }
    else { // modified yen's algorithm
        list<edge*> root_path;
        //uint32_t max_spur_edge = min(m.P_ranking.back()->p->E.size(), l.layers.size() - m.ranking_size);
        uint32_t max_spur_edge = m.P_ranking.back()->p->E.size();
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
                spur_path.insert(spur_path.end(), root_path.begin(), root_path.end());
                m.P_potential_ranking.push_back(path_from_edges(l, m, spur_path));
            }

            root_path.push_front(spur_edge);
            for(auto e : l.E) e->removed = false;
        }

        sort(m.P_potential_ranking.begin(), m.P_potential_ranking.end(), compare_model_path_ptr_func);
        m.P_ranking.push_back(m.P_potential_ranking.back());
        m.P_potential_ranking.pop_back();
        if(m.P_potential_ranking.size() > (l.max_rank - m.ranking_size)){
            auto erase_end = m.P_potential_ranking.end() - (l.max_rank - m.ranking_size - 1);
            m.P_potential_ranking.erase(m.P_potential_ranking.begin(), erase_end);
        }
    }

    m.ranking_size = m.P_ranking.size();
    return m.P_ranking.back();
}


struct v_triple{
    edge* e;
    REAL score;
    v_triple* prev_triple;

    bool operator < (const v_triple& r) const { return score < r.score; }
};

bool compare_v_triple_func(const v_triple l, const v_triple r) { return (l < r); }

template<bool yen>
vector<model_path*> top_k_paths(ltls& l, model& m, uint32_t top_k){

    vector<model_path*> top_paths;
    if(yen) {
        l.max_rank = top_k;
        for (uint32_t k = 0; k < top_k; ++k)
            top_paths.push_back(get_next_top(l, m));
    }
    else {
        vector<vector<v_triple>> v_top;
        v_top.resize(l.v);
        for(uint32_t i = 0; i < l.v; ++i)
            v_top[i].reserve(2 * top_k);
        v_top[l.v_source->v].push_back({nullptr, 0, nullptr});

        for(auto _v : l.V){
            for(uint32_t i = 0; i < _v->e_out; ++i){
                edge *_e = _v->E[i];

                for(size_t j = 0; j < v_top[_v->v].size(); ++j){
                    //float new_length = LOG(EXP(m.e_pred[_e->e].scalar) + EXP(v_top[_v->v][j].score));
                    float new_length = m.e_pred[_e->e].scalar + v_top[_v->v][j].score;
                    v_top[_e->v_in->v].push_back({_e, new_length, &v_top[_v->v][j]});
                }

                sort(v_top[_e->v_in->v].rbegin(), v_top[_e->v_in->v].rend(), compare_v_triple_func);
                if(v_top[_e->v_in->v].size() > top_k)
                    v_top[_e->v_in->v].erase(v_top[_e->v_in->v].begin() + top_k, v_top[_e->v_in->v].end());
            }
        }

        for(auto _v_tripe_sink : v_top[l.v_sink->v]) {
            vector<edge*> path;
            v_triple *_v_tripe = &_v_tripe_sink;
            do {
                path.push_back(_v_tripe->e);
                _v_tripe = _v_tripe->prev_triple;
            }while(path.back()->v_out != l.v_source);

            top_paths.push_back(path_from_edges(l, m, path));
        }

    }

    return top_paths;
}


// ensemble top
//----------------------------------------------------------------------------------------------------------------------

template<bool yen, bool brute>
vector<uint32_t> ta_top_k(ltls& l, uint32_t top_k){

    uint32_t rank = 0;
    unordered_set <uint32_t> seen_labels;
    vector<pair<float, uint32_t>> _top_labels;
    float threshold;

    if(yen) {
        do {
            if (rank >= l.max_rank) break;
            //if(rank == l.k) break; // not really needed
            ++rank;

            unordered_set <uint32_t> new_labels;
            threshold = 0;

            for (uint32_t i = 0; i < l.models_to_use; ++i) {
                model_path *_p = get_next_top(l, l.M[i]);
                threshold += _p->score;

                if (seen_labels.find(_p->label) == seen_labels.end()) {
                    new_labels.insert(_p->label);
                    seen_labels.insert(_p->label);
                }
            }

            for (auto label : new_labels) {
                float label_aggregate_score = 0;
                for (uint32_t i = 0; i < l.models_to_use; ++i)
                    label_aggregate_score += get_label_score(l, l.M[i], label);
                _top_labels.push_back({label_aggregate_score, label});
            }

            if (_top_labels.size() > top_k) {
                sort(_top_labels.rbegin(), _top_labels.rend());
                do {
                    _top_labels.pop_back();
                } while (_top_labels.size() > top_k);
            }

        } while (_top_labels.size() != top_k || _top_labels.back().first < threshold);
    }
    else if(brute){
        for(uint32_t i = 0; i < l.k; ++i){
            _top_labels.push_back({0, i + 1});
            for (uint32_t j = 0; j < l.models_to_use; ++j) {
                update_path(l, l.M[j], l.M[j].P[i].p);
                _top_labels.back().first += l.M[j].P[i].score;
            }
        }
        sort(_top_labels.rbegin(), _top_labels.rend());
    }
    else{

        vector<vector<model_path*>> top_models_paths;
        for (uint32_t i = 0; i < l.models_to_use; ++i)
            top_models_paths.push_back(l.top_k_paths(l, l.M[i], l.max_rank));

        do {
            if (rank >= l.max_rank) break;
            //if(rank == l.k) break; // not really needed
            ++rank;

            unordered_set <uint32_t> new_labels;
            threshold = 0;

            for (uint32_t i = 0; i < l.models_to_use; ++i) {
                model_path *_p = top_models_paths[i][rank - 1];
                threshold += _p->score;

                if (seen_labels.find(_p->label) == seen_labels.end()) {
                    new_labels.insert(_p->label);
                    seen_labels.insert(_p->label);
                }
            }

            for (auto label : new_labels) {
                float label_aggregate_score = 0;
                for (uint32_t i = 0; i < l.models_to_use; ++i)
                    label_aggregate_score += get_label_score(l, l.M[i], label);
                _top_labels.push_back({label_aggregate_score, label});
            }

            if (_top_labels.size() > top_k) {
                sort(_top_labels.rbegin(), _top_labels.rend());
                do {
                    _top_labels.pop_back();
                } while (_top_labels.size() > top_k);
            }

        } while (_top_labels.size() != top_k || _top_labels.back().first < threshold);
    }

    vector<uint32_t> top_labels;
    for(size_t i = 0; i < top_k; ++i)
        top_labels.push_back(_top_labels[i].second);

    l.sum_rank += rank;
    return top_labels;
}

template<bool min>
vector<uint32_t> max_min_top_k(ltls& l, uint32_t top_k){
    unordered_map <uint32_t, float> seen_labels;
    vector<pair<float, uint32_t>> _top_labels;
    vector<uint32_t> top_labels;

    do{
        for(uint32_t i = 0; i < l.models_to_use; ++i) {
            model_path *_p = get_next_top(l, l.M[i]);
            if(seen_labels.find(_p->label) == seen_labels.end())
                seen_labels.insert({_p->label, _p->score});
            else{
                if(min) {
                    if (seen_labels[_p->label] > _p->score)
                        seen_labels[_p->label] = _p->score;
                }
                else{
                    if (seen_labels[_p->label] < _p->score)
                        seen_labels[_p->label] = _p->score;
                }
            }
        }
    } while (seen_labels.size() < l.max_rank);

    for(auto l : seen_labels)
        _top_labels.push_back({l.second, l.first});

    sort(_top_labels.rbegin(), _top_labels.rend());

    for(size_t i = 0; i < top_k; ++i)
        top_labels.push_back(_top_labels[i].second);

    return top_labels;
}

vector<uint32_t> avg_top_k(ltls& l, uint32_t top_k){
    unordered_map <uint32_t, pair<float, uint32_t>> seen_labels;
    vector<pair<float, uint32_t>> _top_labels;
    vector<uint32_t> top_labels;

    for(uint32_t i = 0; i < l.max_rank; ++i) {
        for(uint32_t j = 0; j < l.models_to_use; ++j) {
            model_path *_p = get_next_top(l, l.M[j]);
            if(seen_labels.find(_p->label) == seen_labels.end())
                seen_labels.insert({_p->label, {_p->score, 1}});
            else {
                seen_labels[_p->label].first += _p->score;
                ++seen_labels[_p->label].second;
            }
        }
    }

    for(auto l : seen_labels)
        _top_labels.push_back({l.second.first/l.second.second, l.first});

    sort(_top_labels.rbegin(), _top_labels.rend());

    for(size_t i = 0; i < top_k; ++i)
        top_labels.push_back(_top_labels[i].second);

    return top_labels;
}

template<bool single, bool ta, bool max, bool min, bool avg>
vector<uint32_t> top_k_ensemble(ltls& l, uint32_t top_k){
    if(single){
        vector<model_path*> top_paths = l.top_k_paths(l, l.M[0], top_k);
        vector<uint32_t> top_labels;
        for(auto p : top_paths) top_labels.push_back(p->label);
        return top_labels;
    }
    else if(max)
        return max_min_top_k<false>(l, top_k);
    else if(min)
        return max_min_top_k<true>(l, top_k);
    else if(avg)
        return avg_top_k(l, top_k);
    else //if(ta)
        return l.ta_top_k(l, top_k);
}


// learn
//----------------------------------------------------------------------------------------------------------------------

void add_new_label(ltls& l, uint32_t label) {

    l.seen_labels.insert(label);
    uniform_real_distribution <float> real_dist(0.f, 1.f);

    for(uint32_t i = 0; i < l.ensemble; ++i) {
        model& m = l.M[i];
        path* path_to_assign;
        bool select_randomly = true;

        if(l.best_policy || real_dist(l.rng) < l.mixed_p){ // best assignment
            for(size_t j = 0; j < log2(l.k); ++j){
                auto path_candidate = get_next_top(l, m)->p;
                if(m.available_paths.find(path_candidate) != m.available_paths.end()){
                    path_to_assign = path_candidate;
                    select_randomly = false;
                    continue;
                }
            }
        }

        if(select_randomly){ // random assignment
            uniform_int_distribution <uint32_t> int_dist(0, m.available_paths.size() - 1);
            auto p = int_dist(l.rng);
            path_to_assign = *next(m.available_paths.begin(), p);
        }

        m.available_paths.erase(path_to_assign);
        m.P[label - 1].p = path_to_assign;
    }
}

inline void evaluate(ltls& l, model& m, base_learner& base, example& ec) {
    m.ranking_size = 0;
    ec.l.simple = {FLT_MAX, 0.f, 0.f};
    base.multipredict(ec, m.base, l.e, m.e_pred, false);
    if(SKIPS_MUL) for(size_t i = 0; i < l.e; ++i) m.e_pred[i].scalar *= l.E[i]->skips;
}

void compute_gradients(ltls& l, model& m, uint32_t label){

    for(auto v : l.V) v->alpha = v->beta = numeric_limits<float>::lowest();
    l.v_source->alpha = l.v_sink->beta = 0;

    for(auto e : l.E){
        REAL path_length = e->v_out->alpha + m.e_pred[e->e].scalar;
        e->v_in->alpha = LOG(EXP(e->v_in->alpha) + EXP(path_length));
    }

    for(auto e = l.E.rbegin(); e != l.E.rend(); ++e){
        REAL path_length = (*e)->v_in->beta + m.e_pred[(*e)->e].scalar;
        (*e)->v_out->beta = LOG(EXP((*e)->v_out->beta) + EXP(path_length));
    }

    unordered_set<edge*> positive_edges;
    path *_p = get_path(l, m, label);
    for(auto e : _p->E) positive_edges.insert(e);

    for(auto e : l.E){
        REAL yuv = 0;
        if(positive_edges.find(e) != positive_edges.end()) yuv = 1;
        REAL puvx = EXP(e->v_out->alpha + m.e_pred[e->e].scalar + e->v_in->beta - l.v_sink->alpha);
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

template<bool multilabel>
void learn(ltls& l, base_learner& base, example& ec){

    if(!l.learn_count) l.learn_predict_start_time_point = chrono::steady_clock::now();
    ++l.learn_count;

    COST_SENSITIVE::label ec_labels = ec.l.cs; // copy is needed to restore example later
    if (!ec_labels.costs.size()) return;

    for(uint32_t i = 0; i < l.ensemble; ++i)
        evaluate(l, l.M[i], base, ec);

    uint32_t label;

    if(multilabel){
        uniform_int_distribution <uint32_t> int_dist(0, ec_labels.costs.size() - 1);
        label = ec_labels.costs[int_dist(l.rng)].class_index;
    }
    else
        label = ec_labels.costs[0].class_index;

//    if (l.seen_labels.find(label) == l.seen_labels.end())
//        add_new_label(l, cl.class_index);

    for(uint32_t i = 0; i < l.ensemble; ++i) {
        model& m = l.M[i];
        ec.loss = l.loss_function->loss = 1;
        compute_gradients(l, m, label);
        update_edges(l, m, base, ec);
    }

    ec.l.cs = ec_labels;
    ec.pred.multiclass = 0;
}

// predict
//----------------------------------------------------------------------------------------------------------------------

void l1_const_reg(ltls& l){
    parameters &weights = l.all->weights;
    uint32_t stride_shift = weights.stride_shift();
    uint64_t mask = weights.mask() >> stride_shift;

    for (uint64_t i = 0; i <= mask; ++i) {
        uint64_t idx = (i << stride_shift);
        if(fabs(weights[idx]) < l.l1_const) weights[idx] = 0;
    }
}

void predict(ltls& l, base_learner& base, example& ec){

    if(!l.predict_count){
        if(l.l1_const_reg) l1_const_reg(l);
        l.learn_predict_start_time_point = chrono::steady_clock::now();
    }
    ++l.predict_count;

    COST_SENSITIVE::label ec_labels = ec.l.cs;

    for(uint32_t i = 0; i < l.models_to_use; ++i) evaluate(l, l.M[i], base, ec);
    auto top_paths = l.top_k_ensemble(l, l.p_at_k);

    vector <uint32_t> true_labels;
    for (auto &cl : ec_labels.costs) true_labels.push_back(cl.class_index);

    if (l.p_at_k > 0 && true_labels.size() > 0) {
        for (size_t i = 0; i < l.p_at_k; ++i) {
            if (find(true_labels.begin(), true_labels.end(), top_paths[i]) != true_labels.end())
                l.precision_at_k[i] += 1.0f;
        }
    }

    ec.l.cs = ec_labels;
    ec.pred.multiclass = top_paths.front();
}


// EG
//----------------------------------------------------------------------------------------------------------------------

inline float sigmoid(REAL in) { return 1.0f / (1.0f + exp(-in)); }

inline float path_P(ltls& l, model& m, path *p){
    REAL p_sum = 0;
    for(auto _p : l.P){
        update_path(l, m, _p);
        p_sum += exp(_p->score);
    }

    return exp(p->score)/p_sum;
}

void init_eg(ltls& l){
    l.eg_P = calloc_or_throw<float>(l.models_to_use);
    for(size_t i = 0; i < l.models_to_use; ++i)
        l.eg_P[i] = 1.f/l.models_to_use;

    cerr << "eg_model = " << l.eg_model << endl;

    if(l.predict_eg){
        auto eg_model = ifstream(l.eg_model, ios::binary);
        if(eg_model.good()) {
            uint32_t models_to_use;
            eg_model.read((char*) &models_to_use, sizeof(uint32_t));

            if(models_to_use != l.models_to_use)
                cerr << "Something is wrong with eg_model!\n";

            for(size_t i = 0; i < l.models_to_use; ++i)
                eg_model.read((char*) &l.eg_P[i], sizeof(float));

            eg_model.close();
        }
        else
            cerr << "Eg model not found!\n";

        cerr << "eg_weights = [ ";
        for(size_t i = 0; i < l.models_to_use; ++i) {
            l.M[i].eg_P = l.eg_P[i]/(1.f/l.models_to_use);
            cerr << l.eg_P[i] << " ";
        }
        cerr << "]\n";
    }
}

void learn_eg(ltls& l, base_learner& base, example& ec) {

    float *x;
    x = calloc_or_throw<float>(l.models_to_use);
    float Px = 0;
    float sumxPx = 0;

    if(!l.predict_count) l.learn_predict_start_time_point = chrono::steady_clock::now();
    COST_SENSITIVE::label ec_labels = ec.l.cs;

    float eg_eta = l.all->eta;

    for(size_t i = 0; i < l.models_to_use; ++i){
        model &m = l.M[i];
        evaluate(l, m, base, ec);

        auto p = get_path(l, m, ec_labels.costs[0].class_index);
        //update_path(l, m, p);
        //x[i] = sigmoid(p->score);
        x[i] = path_P(l,m, p);
        Px += l.eg_P[i] * x[i];
    }

    for(size_t i = 0; i < l.models_to_use; ++i)
        sumxPx += l.eg_P[i] * exp(eg_eta * (x[i]/Px));

    for(size_t i = 0; i < l.models_to_use; ++i) {
        l.eg_P[i] = l.eg_P[i] * exp(eg_eta * (x[i] / Px)) / sumxPx;
    }

    ++l.predict_count;
    ec.l.cs = ec_labels;
}


// finish
//----------------------------------------------------------------------------------------------------------------------

void finish_example(vw& all, ltls& l, example& ec) {
    all.sd->update(ec.test_only, 0, ec.weight, ec.num_features);
    VW::finish_example(all, &ec);
}

void end_pass(ltls& l){
    ++l.pass_count;
    cerr << "end of pass " << l.pass_count << endl;

    if(l.pass_count == l.all->numpasses && l.l1_const_reg)
        l1_const_reg(l);
}

void w_stats(ltls& l){
    parameters &weights = l.all->weights;
    uint32_t stride_shift = weights.stride_shift();
    uint64_t mask = weights.mask();

    cerr << "bits = " << l.all->num_bits << endl;
    cerr << "bits_size = " << static_cast<uint64_t>(pow(2, l.all->num_bits) - 1) << endl;
    cerr << "mask = " << mask << endl;
    cerr << "stride_shift = " << stride_shift << endl;
}

void w_ranges_stats(ltls& l){

    double w_avg = 0;
    uint64_t w_count = 0;
    uint64_t w_nonzero = 0;
    float w_ranges[] = {5, 2.5, 1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001, 0, -0.001, -0.005, -0.01, -0.05, -0.1, -0.25, -0.5, -0.75, -1, -2.5, -5, -100};
    uint64_t w_ranges_count[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    parameters &weights = l.all->weights;
    uint32_t stride_shift = weights.stride_shift();
    uint64_t mask = weights.mask() >> stride_shift;

    cerr << "stride_shift = " << stride_shift << endl;
    for (uint64_t i = 0; i <= mask; ++i) {
        uint64_t idx = (i << stride_shift);
        w_avg += weights[idx];
        ++w_count;
        if(weights[idx] != 0) ++w_nonzero;
        for(size_t k = 0; k < 24; ++k) {
            if (weights[idx] >= w_ranges[k]) {
                ++w_ranges_count[k];
                break;
            }
        }
    }

    cerr << "w_avg = " << w_avg/w_count << "\nw_count = " << w_count << "\nw_nonzero = " << w_nonzero << endl;
    for(size_t k = 0; k < 24; ++k)
        cerr << "gt or eq " << w_ranges[k] << " = " << w_ranges_count[k] << endl;
}

void model_size(ltls& l){
    long long model_size1, model_size2;
    model_size1 = model_size2 = l.models_to_use * l.k * sizeof(uint32_t);

    uint32_t predictors_shift = static_cast<uint32_t>(floor(log2(static_cast<float>(l.e * l.ensemble)))) + 1;
    parameters &weights = l.all->weights;
    uint32_t stride_shift = weights.stride_shift();
    uint64_t w_mask = weights.mask() >> stride_shift;
    uint32_t f_mask = w_mask >> predictors_shift;

    cerr << "w_mask = " << w_mask << endl;
    cerr << "f_mask = " << f_mask << endl;

    bool prev_0 = false;
    for (uint64_t i = 0; i <= f_mask; ++i) {
        uint64_t idx = (i << predictors_shift);
        for(uint64_t j = 0; j < l.models_to_use * l.e; ++j){
            size_t w = j << stride_shift;
            if(weights[idx + w] != 0){
                model_size1 += sizeof(float);
                prev_0 = false;
            }
            else if(!prev_0) {
                model_size1 += sizeof(float) + sizeof(uint64_t);
                prev_0 = true;
            }

            model_size2 += sizeof(float);
        }
    }

    cerr << "model_size_features = " << static_cast<float>(model_size1)/1024/1024 << "MB\n";
    cerr << "model_size_uncompressed_features = " << static_cast<float>(model_size2)/1024/1024 << "MB\n";

    model_size1 = model_size2 = l.models_to_use * l.k * sizeof(uint32_t);

    prev_0 = false;
    for(uint64_t j = 0; j < l.models_to_use * l.e; ++j){
        size_t w = j << stride_shift;

        for (uint64_t i = 0; i <= f_mask; ++i) {
            uint64_t idx = (i << predictors_shift);

            if(weights[idx + w] != 0){
                model_size1 += sizeof(float);
                prev_0 = false;
            }
            else if(!prev_0) {
                model_size1 += sizeof(float) + sizeof(uint64_t);
                prev_0 = true;
            }

            model_size2 += sizeof(float);
        }
    }

    cerr << "model_size_predictors = " << static_cast<float>(model_size1)/1024/1024 << "MB\n";
    cerr << "model_size_uncompressed_predictors = " << static_cast<float>(model_size2)/1024/1024 << "MB\n";
}

void finish(ltls& l){
    auto end_time_point = chrono::steady_clock::now();
    auto execution_time = end_time_point - l.learn_predict_start_time_point;
    cerr << "learn_predict_time = " << static_cast<double>(chrono::duration_cast<chrono::microseconds>(execution_time).count()) / 1000000 << "s\n";

    float correct = 0;
    for(size_t i = 0; i < l.p_at_k; ++i) {
        correct += l.precision_at_k[i];
        cerr << "P@" << i + 1 << " = " << correct / (l.predict_count * (i + 1)) << endl;
    }

    cerr << "get_next_top_count = " << l.get_next_top_count << "\nget_path_score_count = " << l.get_path_score_count << endl;
    float average_rank = static_cast<float>(l.sum_rank)/l.predict_count;
    cerr << "average_rank = " << average_rank << endl;

    if(l.stats) {
        w_stats(l);
        w_ranges_stats(l);
        model_size(l);
    }

    if(l.learn_eg){
        cerr << "eg_weights = [ ";
        for(size_t i = 0; i < l.models_to_use; ++i) {
            cerr << l.eg_P[i] << " ";
        }
        cerr << "]\n";

        auto eg_model = ofstream(l.eg_model, ios::binary);
        if(eg_model.good()) {
            eg_model.write((char*) &l.models_to_use, sizeof(uint32_t));
            //eg_model.write((char*) &l.eg_P, l.models_to_use * sizeof(float));
            for(size_t i = 0; i < l.models_to_use; ++i)
                eg_model.write((char*) &l.eg_P[i], sizeof(float));

            eg_model.close();
        }
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

void save_load_graphs(ltls& l, io_buf& model_file, bool read, bool text){

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

//            cout << "MODEL " << i << " = [ ";
//            for(uint32_t p = 0; p < l.k; ++p){
//                cout << m.P[p].label << "(" << m.P[p].p->label << ") ";
//            }
//            cout << "]\n";

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
            ("p_at", po::value<uint32_t>(), "P@k (default 1)")
            ("l1_const", po::value<float>(), "")
            ("stats", "")
            ("multilabel", "")

            // learning policy
            ("inorder_policy", "inorder path assignment policy")
            ("random_policy", "random path assignment policy")
            ("best_policy", "best possible path assignment policy")
            ("mixed_policy", po::value<float>(), "random mixed with best possible path policy")

            // prediction policy
            ("yen", "use yen")
            ("brute", "")
            ("learn_eg", po::value<string>(), "learn eg for ensemble")
            ("predict_eg", po::value<string>(), "predict using eg model for ensemble")
            ("ta_policy", "threshold algorithm policy")
            ("max_policy", "max score from x(max_rank) unique obtained")
            ("min_policy", "min score from x(max_rank) unique obtained")
            ("avg_policy", "avg of scores from top x(max_rank) obtained");
    add_options(all);

    ltls& data = calloc_or_throw<ltls>();

    data.k = all.vm["ltls"].as<size_t>();
    data.ensemble = 1;
    data.random_policy = data.best_policy = false;
    data.all = &all;

    data.precision = 0;
    data.predicted_number = 0;
    data.predict_count = 0;
    data.p_at_k = 1;
    data.pass_count = 0;
    data.l1_const_reg = false;
    data.rng.seed(all.random_seed);

    data.seen_labels = unordered_set<uint32_t>();

    data.stats = false;
    data.get_next_top_count = 0;
    data.get_path_score_count = 0;
    data.predict_count = 0;
    data.learn_count = 0;

    // ltls parse options
    //------------------------------------------------------------------------------------------------------------------

    if(all.vm.count("stats")) data.stats = true;

    learner<ltls> *l;
    string path_assignment_policy = "inorder_policy";

    if(all.vm.count("ensemble")) data.ensemble = all.vm["ensemble"].as<uint32_t>();
    *(all.file_options) << " --ensemble " << data.ensemble;
    data.models_to_use = data.ensemble;

    if(all.vm.count("models_to_use")) data.models_to_use = all.vm["models_to_use"].as<uint32_t>();
    if(data.models_to_use > data.ensemble) data.models_to_use = data.ensemble;
    if(data.models_to_use == 1) data.top_k_ensemble = top_k_ensemble<true, false, false, false, false>;
    else{
        if(all.vm.count("max_policy")) data.top_k_ensemble = top_k_ensemble<false, false, true, false, false>;
        if(all.vm.count("min_policy")) data.top_k_ensemble = top_k_ensemble<false, false, false, true, false>;
        if(all.vm.count("avg_policy")) data.top_k_ensemble = top_k_ensemble<false, false, false, false, true>;
        else //if(all.vm.count("ta_policy"))
            data.top_k_ensemble = top_k_ensemble<false, true, false, false, false>;
    }

    if(all.vm.count("yen")){
        data.top_k_paths = top_k_paths<true>;
        data.ta_top_k = ta_top_k<true, false>;
    }
    else if(all.vm.count("brute")){
        data.top_k_paths = top_k_paths<true>;
        data.ta_top_k = ta_top_k<false, true>;
    }
    else{
        data.top_k_paths = top_k_paths<false>;
        data.ta_top_k = ta_top_k<false, false>;
    }

    if( all.vm.count("p_at") ) data.p_at_k = all.vm["p_at"].as<uint32_t>();
    data.precision_at_k.resize(data.p_at_k);
    data.max_rank = data.p_at_k;
    if( all.vm.count("max_rank") ) data.max_rank = all.vm["max_rank"].as<uint32_t>();
    if(data.max_rank > data.k) data.max_rank = data.k;

    if(all.vm.count("inorder_policy")) {
        *(all.file_options) << " --inorder_policy";
    }
    else if(all.vm.count("best_policy")) {
        path_assignment_policy = "best_policy";
        data.best_policy = true;
    }
    else if(all.vm.count("mixed_policy")) {
        path_assignment_policy = "mixed_policy";
        data.mixed_p = all.vm["mixed_policy"].as<float>();
        data.mixed_policy = true;
    }
    else if(all.vm.count("random_policy") || data.ensemble) {
        path_assignment_policy = "random_policy";
        data.random_policy = true;
    }

    if(all.vm.count("l1_const")) {
        data.l1_const_reg = true;
        data.l1_const = all.vm["l1_const"].as<float>();
        if(data.l1_const == 0) data.l1_const_reg = false;
    }

    data.runtime_path_assignment = data.best_policy || data.mixed_policy;

    // ltls init graph and paths
    //------------------------------------------------------------------------------------------------------------------

    init_graph(data);
    init_paths(data);
    init_models(data);

    data.loss_function = new ltls_loss_function();
    all.loss = data.loss_function;

    if(all.vm.count("learn_eg")) {
        data.learn_eg = true;
        data.eg_model = all.vm["learn_eg"].as<string>();
        init_eg(data);
        l = &init_multiclass_learner(&data, setup_base(all), learn<false>, learn_eg, all.p, data.e * data.ensemble);
    }
    else if(all.vm.count("predict_eg")) {
        data.predict_eg = true;
        data.eg_model = all.vm["predict_eg"].as<string>();
        init_eg(data);
        l = &init_multiclass_learner(&data, setup_base(all), learn<false>, predict, all.p, data.e * data.ensemble);
    }
    else if(all.vm.count("multilabel"))
        l = &init_multiclass_learner(&data, setup_base(all), learn<true>, predict, all.p, data.e * data.ensemble);
    else
        l = &init_multiclass_learner(&data, setup_base(all), learn<false>, predict, all.p, data.e * data.ensemble);

    // override parser
    //------------------------------------------------------------------------------------------------------------------

    all.p->lp = COST_SENSITIVE::cs_label;
    all.cost_sensitive = make_base(*l);

    all.holdout_set_off = true; // turn off stop based on holdout loss


    // log info & add some event handlers
    //------------------------------------------------------------------------------------------------------------------

    cerr << "ltls\nk = " << data.k << "\nv = " << data.v
         << "\ne = " << data.e << "\nlayers = " << data.layers.size() << endl;

    if(data.ensemble > 1) {
        cerr << "ensemble = " << data.ensemble
             << "\nmodels_to_use = " << data.models_to_use
             << "\nmax_rank = " << data.max_rank << endl;
    }

    if(path_assignment_policy.length()) cerr << path_assignment_policy << endl;

    l->set_save_load(save_load_graphs);
    l->set_finish_example(finish_example);
    l->set_end_pass(end_pass);
    l->set_finish(finish);

    return all.cost_sensitive;
}
