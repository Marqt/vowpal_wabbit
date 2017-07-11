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
#include <random>

#include "reductions.h"
#include "vw.h"

using namespace std;
using namespace LEARNER;

#define DEBUG false
#define D_COUT if(DEBUG) cout

struct br {
    vw* all;

    uint32_t k; // number of labels
    uint32_t l; // number of selected labels
    bool filter_labels;
    vector<uint32_t> selected_labels;
    uint32_t predict_count;
    
    float inner_threshold;  // inner threshold
    bool positive_labels;   // print positive labels
    bool top_k_labels;   // print top-k labels

    uint32_t p_at_k;
    float precision;
    float predicted_number;
    vector<float> precision_at_k;
    size_t prediction_count;

    uint32_t ec_count;
    chrono::time_point<chrono::steady_clock> learn_predict_start_time_point;
};

// debug helpers - to delete
//----------------------------------------------------------------------------------------------------------------------


// helpers
//----------------------------------------------------------------------------------------------------------------------

inline float sigmoid(float in) { return 1.0f / (1.0f + exp(-in)); }

vector<string> split(string text, char d = ' '){
    vector<string> result;
    const char *str = text.c_str();

    do {
        const char *begin = str;
        while(*str != d && *str) ++str;
        result.push_back(string(begin, str));
    } while (0 != *str++);

    return result;
}


// learn
//----------------------------------------------------------------------------------------------------------------------

template<bool filter_labels>
void learn(br& b, base_learner& base, example& ec){

    COST_SENSITIVE::label ec_labels = ec.l.cs;
    double t = b.all->sd->t;
    double weighted_holdout_examples = b.all->sd->weighted_holdout_examples;
    b.all->sd->weighted_holdout_examples = 0;

    ec.l.simple = {1.f, 0.f, 0.f};
    if(filter_labels) {
        if (ec_labels.costs.size() > 0) {
            for (int i = 0; i < b.selected_labels.size(); ++i) {
                float label = -1.0;
                for (auto &cl : ec_labels.costs) {
                    if (cl.class_index == b.selected_labels[i]) {
                        label = 1.0;
                        break;
                    }
                }

                ec.l.simple.label = label;
                base.learn(ec, i);
            }
        }
    }
    else {
        if (ec_labels.costs.size() > 0) {
            for (int i = 0; i < b.k; ++i) {
                float label = -1.0;
                for (auto &cl : ec_labels.costs) {
                    if (cl.class_index == i) {
                        label = 1.0;
                        break;
                    }
                }

                ec.l.simple.label = label;
                base.learn(ec, i);
            }
        }
    }

    ec.l.cs = ec_labels;
    b.all->sd->t = t + 1;
    b.all->sd->weighted_holdout_examples = weighted_holdout_examples;
    ec.pred.multiclass = 0;

    D_COUT << "----------------------------------------------------------------------------------------------------\n";
}


// predict
//----------------------------------------------------------------------------------------------------------------------

template<bool filter_labels>
void predict(br& b, base_learner& base, example& ec){

    COST_SENSITIVE::label ec_labels = ec.l.cs;
    ++b.predict_count;

    vector<pair<float, uint32_t>> predicted_labels;
    for(uint32_t i = 0; i < b.l; ++i){
        ec.l.simple = {FLT_MAX, 0.f, 0.f};
        base.predict(ec, i);
        if(filter_labels)
            predicted_labels.push_back({ec.pred.scalar, b.selected_labels[i]});
        else
            predicted_labels.push_back({ec.pred.scalar, i});
    }

    sort(predicted_labels.rbegin(), predicted_labels.rend());

    if (b.p_at_k > 0 && ec_labels.costs.size() > 0) {

        vector<uint32_t> true_labels;
        for (auto &cl : ec_labels.costs) true_labels.push_back(cl.class_index);

        if (b.p_at_k > 0 && true_labels.size() > 0) {
            for (size_t i = 0; i < b.p_at_k; ++i) {
                if (find(true_labels.begin(), true_labels.end(), predicted_labels[i].second) != true_labels.end())
                    b.precision_at_k[i] += 1.0f;
            }
        }
    }

    ec.l.cs = ec_labels;
}

void finish_example(vw& all, br& b, example& ec){
    all.sd->update(ec.test_only, 0.0f, ec.weight, ec.num_features);
    VW::finish_example(all, &ec);
}

void pass_end(br& b){
    cout << "end of pass " << b.all->passes_complete << "\n";
}

void finish(br& b){
//    auto end_time_point = chrono::steady_clock::now();
//    auto execution_time = end_time_point - l.learn_predict_start_time_point;
//    cerr << "learn_predict_time = " << static_cast<double>(chrono::duration_cast<chrono::microseconds>(execution_time).count()) / 1000000 << "s\n";

    float correct = 0;
    for(size_t i = 0; i < b.p_at_k; ++i) {
        correct += b.precision_at_k[i];
        cerr << "P@" << i + 1 << " = " << correct / (b.predict_count * (i + 1)) << endl;
    }
}

// setup
//----------------------------------------------------------------------------------------------------------------------

base_learner* br_setup(vw& all) //learner setup
{
    if (missing_option<size_t, true>(all, "br", "Use binary relevance for multilabel with <k> labels"))
        return nullptr;
    new_options(all, "br options")
            ("filter_labels", po::value<string>(), "learn only selected labels")
            ("p_at", po::value<uint32_t>(), "P@k (default 1)");
    add_options(all);

    br& data = calloc_or_throw<br>();
    data.k = (uint32_t)all.vm["br"].as<size_t>();
    data.l = data.k;
    data.filter_labels = false;
    data.predict_count = 0;
    data.all = &all;

    data.p_at_k = 1;
    learner<br> *l;

    if( all.vm.count("p_at") )
        data.p_at_k = all.vm["p_at"].as<uint32_t>();
    data.precision_at_k.resize(data.p_at_k);

    if( all.vm.count("filter_labels")){
        data.filter_labels = true;
        data.selected_labels.clear();

        string labels_str = all.vm["filter_labels"].as<string>();
        *(all.file_options) << " --filter_labels " << labels_str;
        auto labels_vec = split(labels_str, ',');
        for(auto l : labels_vec)
            data.selected_labels.push_back(stoi(l));
        data.l = data.selected_labels.size();
    }

    // init multiclass learner
    // -----------------------------------------------------------------------------------------------------------------

    if(data.filter_labels)
        l = &init_multiclass_learner(&data, setup_base(all), learn<true>, predict<true>, all.p, data.l);
    else
        l = &init_multiclass_learner(&data, setup_base(all), learn<false>, predict<false>, all.p, data.l);
    l->set_finish(finish);

    // override parser
    //------------------------------------------------------------------------------------------------------------------
    all.p->lp = COST_SENSITIVE::cs_label;
    all.cost_sensitive = make_base(*l);

    // log info & add some event handlers
    //------------------------------------------------------------------------------------------------------------------
    cout << "br\nk = " << data.l << "/" << data.k << endl;
    if(data.filter_labels) {
        cout << "selected_labels =";
        for (auto l : data.selected_labels)
            cout << " " << l;
        cout << endl;
    }

    all.holdout_set_off = true; // turn off stop based on holdout loss

    l->set_finish_example(finish_example);
    l->set_end_pass(pass_end);
    l->set_finish(finish);

    return all.cost_sensitive;
}
