#pragma once
#include"type_def.h"

template<size_t VALUE_SIZE, size_t INDEX_SIZE>
struct graph_node {
    value_t values[VALUE_SIZE];
    idx_t indexes[INDEX_SIZE];
};