// Dynamic eval — JSON spec → AST → integer centipawn score.
//
// Hand-rolled minimal JSON parser tailored to the strategy schema. Recognises
// object/array/string/number/null. No third-party deps.

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "eval_dyn.h"
#include "features.h"

namespace math_engine::eval_dyn {

namespace {

// ---------------- Tiny JSON parser (recursive descent) ----------------

struct JsonValue;
using JsonObject = std::vector<std::pair<std::string, std::shared_ptr<JsonValue>>>;
using JsonArray  = std::vector<std::shared_ptr<JsonValue>>;

enum class JT { OBJECT, ARRAY, STRING, NUMBER, BOOLEAN, NIL };

struct JsonValue {
    JT tag;
    JsonObject obj;
    JsonArray  arr;
    std::string str;
    double num = 0.0;
    bool boolean = false;
};

struct Parser {
    const std::string& src;
    size_t pos = 0;
    explicit Parser(const std::string& s) : src(s) {}

    [[noreturn]] void error(const std::string& msg) {
        throw std::runtime_error("eval_dyn JSON parse error at byte " +
                                 std::to_string(pos) + ": " + msg);
    }

    void skip_ws() {
        while (pos < src.size() && std::isspace(static_cast<unsigned char>(src[pos]))) ++pos;
    }

    char peek() {
        skip_ws();
        if (pos >= src.size()) error("unexpected end of input");
        return src[pos];
    }

    void expect(char c) {
        skip_ws();
        if (pos >= src.size() || src[pos] != c) {
            error(std::string("expected '") + c + "'");
        }
        ++pos;
    }

    std::shared_ptr<JsonValue> parse_value() {
        skip_ws();
        if (pos >= src.size()) error("expected value");
        char c = src[pos];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string_val();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        error(std::string("unexpected character '") + c + "'");
    }

    std::shared_ptr<JsonValue> parse_object() {
        auto v = std::make_shared<JsonValue>();
        v->tag = JT::OBJECT;
        expect('{');
        skip_ws();
        if (pos < src.size() && src[pos] == '}') { ++pos; return v; }
        for (;;) {
            std::string key = parse_string_raw();
            expect(':');
            auto val = parse_value();
            v->obj.emplace_back(std::move(key), std::move(val));
            skip_ws();
            if (pos < src.size() && src[pos] == ',') { ++pos; continue; }
            expect('}');
            return v;
        }
    }

    std::shared_ptr<JsonValue> parse_array() {
        auto v = std::make_shared<JsonValue>();
        v->tag = JT::ARRAY;
        expect('[');
        skip_ws();
        if (pos < src.size() && src[pos] == ']') { ++pos; return v; }
        for (;;) {
            v->arr.push_back(parse_value());
            skip_ws();
            if (pos < src.size() && src[pos] == ',') { ++pos; continue; }
            expect(']');
            return v;
        }
    }

    std::string parse_string_raw() {
        skip_ws();
        expect('"');
        std::string out;
        while (pos < src.size() && src[pos] != '"') {
            if (src[pos] == '\\' && pos + 1 < src.size()) {
                char e = src[pos + 1];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    default:  out += e;   break;   // best-effort
                }
                pos += 2;
            } else {
                out += src[pos++];
            }
        }
        expect('"');
        return out;
    }

    std::shared_ptr<JsonValue> parse_string_val() {
        auto v = std::make_shared<JsonValue>();
        v->tag = JT::STRING;
        v->str = parse_string_raw();
        return v;
    }

    std::shared_ptr<JsonValue> parse_number() {
        size_t start = pos;
        if (src[pos] == '-') ++pos;
        while (pos < src.size() && (std::isdigit(static_cast<unsigned char>(src[pos]))
                                     || src[pos] == '.' || src[pos] == 'e'
                                     || src[pos] == 'E' || src[pos] == '+' || src[pos] == '-')) {
            ++pos;
        }
        auto v = std::make_shared<JsonValue>();
        v->tag = JT::NUMBER;
        v->num = std::strtod(src.c_str() + start, nullptr);
        return v;
    }

    std::shared_ptr<JsonValue> parse_bool() {
        auto v = std::make_shared<JsonValue>();
        v->tag = JT::BOOLEAN;
        if (src.compare(pos, 4, "true") == 0)  { pos += 4; v->boolean = true;  return v; }
        if (src.compare(pos, 5, "false") == 0) { pos += 5; v->boolean = false; return v; }
        error("invalid literal");
    }

    std::shared_ptr<JsonValue> parse_null() {
        if (src.compare(pos, 4, "null") == 0) {
            auto v = std::make_shared<JsonValue>();
            v->tag = JT::NIL;
            pos += 4;
            return v;
        }
        error("invalid literal");
    }
};

const JsonValue* object_get(const JsonValue& v, const std::string& key) {
    if (v.tag != JT::OBJECT) return nullptr;
    for (const auto& kv : v.obj) {
        if (kv.first == key) return kv.second.get();
    }
    return nullptr;
}

// ---------------- AST builder ----------------

std::unique_ptr<ast::Node> build_tree(const JsonValue& v) {
    if (v.tag != JT::OBJECT) {
        throw std::runtime_error("eval_dyn: expression node must be a JSON object");
    }
    if (const auto* var = object_get(v, "var")) {
        if (var->tag != JT::STRING) throw std::runtime_error("eval_dyn: 'var' must be a string");
        int idx = features::feature_index(var->str);
        if (idx < 0) throw std::runtime_error("eval_dyn: unknown feature variable: " + var->str);
        return ast::make_var(idx);
    }
    if (const auto* c = object_get(v, "const")) {
        if (c->tag != JT::NUMBER) throw std::runtime_error("eval_dyn: 'const' must be a number");
        return ast::make_const(static_cast<float>(c->num));
    }
    const auto* op_v = object_get(v, "op");
    if (op_v == nullptr || op_v->tag != JT::STRING) {
        throw std::runtime_error("eval_dyn: node missing var/const/op");
    }
    const auto* args_v = object_get(v, "args");
    if (args_v == nullptr || args_v->tag != JT::ARRAY) {
        throw std::runtime_error("eval_dyn: op node missing args array");
    }
    const std::string& op_name = op_v->str;
    ast::BinOp bop;
    ast::UnOp  uop;
    if (ast::parse_bin_op(op_name, bop)) {
        if (args_v->arr.size() != 2) {
            throw std::runtime_error("eval_dyn: binop '" + op_name + "' needs 2 args");
        }
        return ast::make_binop(bop,
            build_tree(*args_v->arr[0]),
            build_tree(*args_v->arr[1]));
    }
    if (ast::parse_un_op(op_name, uop)) {
        if (args_v->arr.size() != 1) {
            throw std::runtime_error("eval_dyn: unop '" + op_name + "' needs 1 arg");
        }
        return ast::make_unop(uop, build_tree(*args_v->arr[0]));
    }
    throw std::runtime_error("eval_dyn: unknown operator: " + op_name);
}

std::string slurp(const std::string& path) {
    // Use C FILE* — std::ifstream segfaults in some MinGW configs when linked
    // into the larger engine library.
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr) {
        throw std::runtime_error("eval_dyn: cannot open strategy file: " + path);
    }
    std::fseek(fp, 0, SEEK_END);
    long n = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (n < 0) { std::fclose(fp); throw std::runtime_error("eval_dyn: ftell failed"); }
    std::string out(static_cast<size_t>(n), '\0');
    if (n > 0) {
        size_t r = std::fread(out.data(), 1, static_cast<size_t>(n), fp);
        if (r != static_cast<size_t>(n)) {
            std::fclose(fp);
            throw std::runtime_error("eval_dyn: short read on " + path);
        }
    }
    std::fclose(fp);
    return out;
}

}  // namespace

StrategySpec load_strategy(const std::string& json_path) {
    std::string text = slurp(json_path);
    Parser p(text);
    auto root_json = p.parse_value();
    if (root_json->tag != JT::OBJECT) {
        throw std::runtime_error("eval_dyn: strategy JSON root must be an object");
    }

    const auto* tree_v = object_get(*root_json, "expression_tree");
    if (tree_v == nullptr) {
        throw std::runtime_error(
            "eval_dyn: strategy JSON missing 'expression_tree' field");
    }

    StrategySpec spec;
    if (const auto* id_v = object_get(*root_json, "id")) {
        if (id_v->tag == JT::STRING) spec.id = id_v->str;
    }
    if (spec.id.empty()) spec.id = "anonymous";

    spec.root = build_tree(*tree_v);

    if (const auto* scale_v = object_get(*root_json, "output_scale")) {
        if (scale_v->tag == JT::NUMBER) spec.output_scale = static_cast<float>(scale_v->num);
    }
    return spec;
}

int evaluate(const StrategySpec& spec, const chess::Board& board) {
    auto features = features::compute_features(board);
    float val = ast::eval_node(*spec.root, features);
    val *= spec.output_scale;
    if (val >  30000.0f) val =  30000.0f;
    if (val < -30000.0f) val = -30000.0f;
    return static_cast<int>(val);
}

}  // namespace math_engine::eval_dyn
