// Chess feature bank — board → feature vector.
//
// The feature catalog must match the Python chess_feature_bank module 1:1.
// Each feature name has a fixed canonical index used both for serialization
// (strategy JSON references features by NAME) and runtime (the eval reads
// the precomputed vector by INDEX).
//
// To add a feature: append it to FEATURE_NAMES in features.cpp, implement the
// computation in compute_features, and add the matching extractor on the
// Python side.

#pragma once

#include "chess.hpp"

#include <string>
#include <vector>

namespace math_engine::features {

// Canonical ordered feature names. Position in this vector IS the index.
const std::vector<std::string>& feature_names();

// Returns -1 if name is unknown.
int feature_index(const std::string& name);

// Compute every feature for the given board, in canonical order.
// Output vector has size feature_names().size().
std::vector<float> compute_features(const chess::Board& board);

}  // namespace math_engine::features
