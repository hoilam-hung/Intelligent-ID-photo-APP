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
	public��
		/ /һϵ�к���...
		/*
		flag �����а�����Ź��ھ������Ϣ����:
			-Mat �ı�ʶ
			-�����Ƿ�����
			-���
			-ͨ����Ŀ
			*/
		int flags;

	int dims;//!�����ά����ȡֵ���ڵ���2//!�к��е�������������󳬹� 2 ά����������ֵΪ-1

	int rows, cols;

	uchar *data;//!ָ�����ݵ�ָ��

	int * refcount;//!ָ������ü����� ��
	/ / ����ָ���û����������ʱ����ָ��Ϊ NULL

		/ / ������Ա
		...
};
