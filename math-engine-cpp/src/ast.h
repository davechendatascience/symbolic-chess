// AST node types for dynamic (JSON-loaded) symbolic eval.
//
// A Node is one of:
//   VAR        — references a feature by index into the feature vector
//   CONST      — a float literal
//   BINOP(op)  — two children, binary operator
//   UNOP(op)   — one child, unary operator
//
// Operator set matches the PySR alphabet plus safe-div semantics:
//   binary: +, -, *, /, min, max
//   unary : tanh, abs, sign, neg
//
// All evaluation happens in float; the dispatcher in eval.cpp truncates to int.
// Division uses safe-div (zero denominator returns 0) to match Python's _safe_div.

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace math_engine::ast {

enum class NodeTag { VAR, CONST, BINOP, UNOP };

enum class BinOp { ADD, SUB, MUL, DIV, MIN, MAX };
enum class UnOp  { TANH, ABS, SIGN, NEG };

struct Node {
    NodeTag tag;

    // VAR
    int var_index = -1;

    // CONST
    float const_val = 0.0f;

    // BINOP / UNOP
    BinOp bin_op{};
    UnOp  un_op{};
    std::unique_ptr<Node> child_a;   // first arg (BINOP/UNOP)
    std::unique_ptr<Node> child_b;   // second arg (BINOP only)
};

// Helpers
inline std::unique_ptr<Node> make_var(int idx) {
    auto n = std::make_unique<Node>();
    n->tag = NodeTag::VAR;
    n->var_index = idx;
    return n;
}
inline std::unique_ptr<Node> make_const(float v) {
    auto n = std::make_unique<Node>();
    n->tag = NodeTag::CONST;
    n->const_val = v;
    return n;
}
inline std::unique_ptr<Node> make_binop(BinOp op,
                                        std::unique_ptr<Node> a,
                                        std::unique_ptr<Node> b) {
    auto n = std::make_unique<Node>();
    n->tag = NodeTag::BINOP;
    n->bin_op = op;
    n->child_a = std::move(a);
    n->child_b = std::move(b);
    return n;
}
inline std::unique_ptr<Node> make_unop(UnOp op, std::unique_ptr<Node> a) {
    auto n = std::make_unique<Node>();
    n->tag = NodeTag::UNOP;
    n->un_op = op;
    n->child_a = std::move(a);
    return n;
}

// String → enum lookup. Returns false on unknown name.
bool parse_bin_op(const std::string& name, BinOp& out);
bool parse_un_op (const std::string& name, UnOp&  out);

// Evaluate a node against a feature vector (sized to feature_bank.size()).
// safe_div: division by zero returns 0 (matches Python _safe_div).
float eval_node(const Node& n, const std::vector<float>& features);

// Pretty-print a node (debugging / cross-check).
std::string to_string(const Node& n,
                      const std::vector<std::string>& feature_names);

}  // namespace math_engine::ast
