#ifndef SORT_H
#define SORT_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

namespace ldpc::sort {
    struct str {
        double value;
        int index;
    };

// Factorial function:
// http://www.rosettacode.org/wiki/Factorial#C
    unsigned long long
    ncr(int n, int k) {
        if (k > n) {
            return 0;
        }
        unsigned long long r = 1;
        for (unsigned long long d = 1; d <= k; ++d) {
            r *= n--;
            r /= d;
        }
        return r;
    }

//https://stackoverflow.com/a/36714204
//Modified slighty to sort indices in ascending order.



    int cmp(const void *a, const void *b) {
        auto *a1 = (struct str *) a;
        auto *a2 = (struct str *) b;
        if ((*a1).value > (*a2).value) {
            return 1;
        }
        if ((*a1).value < (*a2).value) {
            return -1;
        }
        return 0;
    }


    // decoder-bench fork (docs/ARCHITECTURE.md section 9.1): the column order is ascending soft
    // decision, ties broken by ascending column index, with NaN after every number (NaNs by column
    // index). Upstream used qsort with a comparator that made NaN incomparable, so the result
    // depended on the sort implementation. The comparator below is a strict weak order, which makes
    // the sorted sequence unique; the buffer is a vector so a shot allocates nothing here either.
    void soft_decision_col_sort(std::vector<double> &soft_decisions, std::vector<int> &cols, int N) {
        std::vector<struct str> objects(static_cast<size_t>(N > 0 ? N : 0));
        for (int i = 0; i < N; i++) {
            objects[i].value = soft_decisions[i];
            objects[i].index = i;
        }
        // sort objects by (NaN last, value ascending, index ascending)
        std::stable_sort(objects.begin(), objects.end(), [](const struct str &a, const struct str &b) {
            const bool a_nan = std::isnan(a.value);
            const bool b_nan = std::isnan(b.value);
            if (a_nan != b_nan) return !a_nan;
            if (a_nan) return a.index < b.index;
            if (a.value < b.value) return true;
            if (b.value < a.value) return false;
            return a.index < b.index;
        });
        for (int i = 0; i < N; i++)
            cols[i] = objects[i].index;
    }


//https://stackoverflow.com/a/36714204
//Integer sort version Modified slighty to sort indices in ascending order.

} //end namespace ldpc::sort


#endif