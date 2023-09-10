#ifndef HOUGH_IMAGES_H
#define HOUGH_IMAGES_H

#include <vector>
using namespace std;

namespace transformImages {

	class Hough_I {
	public:
		Hough_I();
		virtual ~Hough_I();
	public:
		void Transform(vector<unsigned char*> & images, vector<pair<int, int>> dimensions, size_t num_images);
		vector< pair< pair<int, int>, pair<int, int> > > GetLines(int threshold, int pos);
		const unsigned int* GetAccu(int* w, int* h, int pos);
	private:
		vector<unsigned int *> accu;
		vector<int> accu_w;
		vector<int> accu_h;
		vector<int> img_w;
		vector<int> img_h;
		vector<double> hough_h;
		int num_img;
	};

}

#endif /* HOUGH_IMAGES_H */