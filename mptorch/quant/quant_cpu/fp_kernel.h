#pragma once
#include "modes.h"

float cast_fp_nearest_even(float origin_float, int man_bits, int exp_bits,
													 int bias, bool saturate, SubnormalsMode subnormals);

float cast_fp_nearest_away(float origin_float, int man_bits, int exp_bits,
													 int bias, bool saturate, SubnormalsMode subnormals);

float cast_fp_up(float origin_float, int man_bits, int exp_bits, int bias,
								 bool saturate, SubnormalsMode subnormals);

float cast_fp_down(float origin_float, int man_bits, int exp_bits, int bias,
									 bool saturate, SubnormalsMode subnormals);

float cast_fp_zero(float origin_float, int man_bits, int exp_bits, int bias,
									 bool saturate, SubnormalsMode subnormals);

float cast_fp_stochastic(float origin_float, int man_bits, int exp_bits,
												 int bias, bool saturate, SubnormalsMode subnormals);

float cast_fp_stochastic(float origin_float, int man_bits, int exp_bits,
												 int prng_bits, int bias, bool saturate,
												 SubnormalsMode subnormals);

void fp_kernel(
		float *a, float *o, int size, int man_bits, int exp_bits, int bias, int prng_bits,
		bool saturate = false,
		RoundMode round_mode = RoundMode::RNE,
		SubnormalsMode subnormals_mode = SubnormalsMode::SUBNORMALS);