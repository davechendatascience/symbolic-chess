// AST evaluator + operator-name lookup.

#include "ast.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace math_engine::ast {

bool parse_bin_op(const std::string& name, BinOp& out) {
    if (name == "add" || name == "+")  { out = BinOp::ADD; return true; }
    if (name == "sub" || name == "-")  { out = BinOp::SUB; return true; }
    if (name == "mul" || name == "*")  { out = BinOp::MUL; return true; }
    if (name == "div" || name == "/")  { out = BinOp::DIV; return true; }
    if (name == "min")                 { out = BinOp::MIN; return true; }
    if (name == "max")                 { out = BinOp::MAX; return true; }
    return false;
}

bool parse_un_op(const std::string& name, UnOp& out) {
    if (name == "tanh") { out = UnOp::TANH; return true; }
    if (name == "abs")  { out = UnOp::ABS;  return true; }
    if (name == "sign") { out = UnOp::SIGN; return true; }
    if (name == "neg")  { out = UnOp::NEG;  return true; }
    return false;
}

float eval_node(const Node& n, const std::vector<float>& features) {
    switch (n.tag) {
        case NodeTag::VAR: {
            int idx = n.var_index;
            if (idx < 0 || idx >= static_cast<int>(features.size())) {
                return 0.0f;   // out-of-range feature: treat as 0
            }
            return features[idx];
        }
        case NodeTag::CONST:
            return n.const_val;
        case NodeTag::BINOP: {
            float a = eval_node(*n.child_a, features);
            float b = eval_node(*n.child_b, features);
            switch (n.bin_op) {
                case BinOp::ADD: return a + b;
                case BinOp::SUB: return a - b;
                case BinOp::MUL: return a * b;
                case BinOp::DIV: return (b == 0.0f) ? 0.0f : a / b;
                case BinOp::MIN: return (a < b) ? a : b;
                case BinOp::MAX: return (a > b) ? a : b;
            }
            return 0.0f;
        }
        case NodeTag::UNOP: {
            float x = eval_node(*n.child_a, features);
            switch (n.un_op) {
                case UnOp::TANH: return std::tanh(x);
                case UnOp::ABS:  return std::fabs(x);
                case UnOp::SIGN: return (x > 0.0f) - (x < 0.0f);
                case UnOp::NEG:  return -x;
            }
            return 0.0f;
        }
    }
    return 0.0f;
}

namespace {
const char* bin_name(BinOp op) {
    switch (op) {
        case BinOp::ADD: return "+";
        case BinOp::SUB: return "-";
        case BinOp::MUL: return "*";
        case BinOp::DIV: return "/";
        case BinOp::MIN: return "min";
        case BinOp::MAX: return "max";
    }
    return "?";
}
const char* un_name(UnOp op) {
    switch (op) {
        case UnOp::TANH: return "tanh";
        case UnOp::ABS:  return "abs";
        case UnOp::SIGN: return "sign";
        case UnOp::NEG:  return "neg";
    }
    return "?";
}
}  // namespace

std::string to_string(const Node& n, const std::vector<std::string>& feature_names) {
    std::ostringstream oss;
    switch (n.tag) {
        case NodeTag::VAR:
            if (n.var_index >= 0 && n.var_index < static_cast<int>(feature_names.size()))
                oss << feature_names[n.var_index];
            else
                oss << "var" << n.var_index;
            break;
        case NodeTag::CONST:
            oss << n.const_val;
            break;
        case NodeTag::BINOP:
            oss << "(" << to_string(*n.child_a, feature_names)
                << " " << bin_name(n.bin_op) << " "
                << to_string(*n.child_b, feature_names) << ")";
            break;
        case NodeTag::UNOP:
            oss << un_name(n.un_op) << "("
                << to_string(*n.child_a, feature_names) << ")";
            break;
    }
    return oss.str();
}

}  // namespace math_engine::ast
