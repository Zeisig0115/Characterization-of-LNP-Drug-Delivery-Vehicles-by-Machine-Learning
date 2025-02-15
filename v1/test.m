clear;
clc;

file1 = "./type_1_model/d20.mat";
file2 = "./type_1_model/d100.mat";

[q_1, Iq_1] = rot_modified(file2, 400);
[q_2, Iq_2] = cal(file2, 400);

