#ifndef LINE_FUNCTIONS_H
#define LINE_FUNCTIONS_H

#include <vector>
using namespace std;

int FindIntersectionPoint(pair< pair<int, int>, pair<int, int> > points_l1, pair< pair<int, int>, pair<int, int> > points_l2, pair<int, int>& inpoint);
vector<pair<int, int> > GetIntersectionPoints(vector< pair< pair<int, int>, pair<int, int> > > lines);

#endif
