#pragma once
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class CV_EXPORTS Mat
{
	public：
		/ /一系列函数...
		/*
		flag 参数中包含序号关于矩阵的信息，如:
			-Mat 的标识
			-数据是否连续
			-深度
			-通道数目
			*/
		int flags;

	int dims;//!数组的维数，取值大于等于2//!行和列的数量，如果矩阵超过 2 维，那这两个值为-1

	int rows, cols;

	uchar *data;//!指向数据的指针

	int * refcount;//!指针的引用计数器 ；
	/ / 阵列指向用户分配的数据时，当指针为 NULL

		/ / 其他成员
		...
};
