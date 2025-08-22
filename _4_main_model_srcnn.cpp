/*
#include "ImageLib.h"
#include "CModel.h"
#include <ctime>    // clock()


using namespace std;

// 정확하게 동작시 20점 (부분점수 없음)

int main() {
	Model model;

	// build model
	model.add_layer(new Layer_Conv("Conv1", 9, 1, 64, LOAD_INIT, "C:/Users/82107/Desktop/ParallelProgramming/ParallelProgramming/model/weights_conv1_9x9x1x64.txt", "C:/Users/82107/Desktop/ParallelProgramming/ParallelProgramming/model/biases_conv1_64.txt"));
	model.add_layer(new Layer_ReLU("Relu1", 1, 64, 64));
	model.add_layer(new Layer_Conv("Conv2", 5, 64, 32, LOAD_INIT, "C:/Users/82107/Desktop/ParallelProgramming/ParallelProgramming/model/weights_conv2_5x5x64x32.txt", "C:/Users/82107/Desktop/ParallelProgramming/ParallelProgramming/model/biases_conv2_32.txt"));
	model.add_layer(new Layer_ReLU("Relu2", 1, 32, 32));
	model.add_layer(new Layer_Conv("Conv3", 5, 32, 1, LOAD_INIT, "C:/Users/82107/Desktop/ParallelProgramming/ParallelProgramming/model/weights_conv3_5x5x32x1.txt", "C:/Users/82107/Desktop/ParallelProgramming/ParallelProgramming/model/biases_conv3_1.txt"));


	clock_t T0 = clock();
	model.test("baby_512x512_input.bmp", "baby_512x512_output_srcnn.bmp");
	clock_t T1 = clock();

	double ms = 1000.0 * (T1 - T0) / CLOCKS_PER_SEC;
	std::cout << "TOTAL = " << ms << " ms" << endl;



	model.print_layer_info();
	model.print_tensor_info();
	system("PAUSE");

	return 0;
}
*/
