#include <cmath>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <utility>
#include <vector>
#include "line_functions.h"

using namespace std;

int FindIntersectionPoint(pair< pair<int, int>, pair<int, int> > points_l1, pair< pair<int, int>, pair<int, int> > points_l2, pair<int, int>& inpoint)
{
	float m_l1 = ((float)points_l1.second.second - (float)points_l1.first.second) / ((float)points_l1.second.first - (float)points_l1.first.first);
	float n_l1 = points_l1.first.second - m_l1 * points_l1.first.first;

	float m_l2 = ((float)points_l2.second.second - (float)points_l2.first.second) / ((float)points_l2.second.first - (float)points_l2.first.first);
	float n_l2 = (float)points_l2.first.second - m_l2 * (float)points_l2.first.first;

	if (m_l1 == m_l2)
		return 1;
	float aux = (n_l2 - n_l1) / (m_l1 - m_l2);
	inpoint.first = aux;
	inpoint.second = m_l1 * aux + n_l1;
	return 0;
}

vector<pair<int, int> > GetIntersectionPoints(vector< pair< pair<int, int>, pair<int, int> > > lines)
{
	vector<pair<int, int>> intersectionPoints;

	for (size_t i = 0; i < lines.size(); ++i) {
		for (size_t j = i + 1; j < lines.size(); ++j) {
			pair<int, int> intersection;
			if (FindIntersectionPoint(lines[i], lines[j], intersection) == 0) {
				intersectionPoints.push_back(intersection);
			}
		}
	}

	return intersectionPoints;
}

//int parametricIntersect(float r1, float t1, float r2, float t2, int* x, int* y) {
//	float ct1 = cosf(t1);     //matrix element a
//	float st1 = sinf(t1);     //b
//	float ct2 = cosf(t2);     //c
//	float st2 = sinf(t2);     //d
//	float d = ct1 * st2 - st1 * ct2;        //determinative (rearranged matrix for inverse)
//	if (d != 0.0f) {
//		*x = (int)((st2 * r1 - st1 * r2) / d);
//		*y = (int)((-ct2 * r1 + ct1 * r2) / d);
//		return(1);
//	}
//	else { //lines are parallel and will NEVER intersect!
//		return(0);
//	}
//}