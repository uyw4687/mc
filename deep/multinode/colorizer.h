#pragma once

void ColorizerInit();

void Colorize(float* input, float* network, float* output, int N);

void ColorizerFinalize();
