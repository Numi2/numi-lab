#include "numi_mjcf_import.hpp"

#include <libxml/parser.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <locale>
#include <map>
#include <numbers>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace numi {
namespace {

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;

std::string_view tag(const xmlNode* node) {
    return node->name ? reinterpret_cast<const char*>(node->name) : "XML";
}

[[noreturn]] void invalid(const xmlNode* node, const std::string& reason) {
    throw std::runtime_error("MJCF line " + std::to_string(xmlGetLineNo(node)) +
        " <" + std::string(tag(node)) + ">: " + reason);
}

std::optional<std::string> property(const xmlNode* node, const char* key) {
    auto* value = xmlGetProp(node, BAD_CAST key);
    if (value == nullptr) return std::nullopt;
    std::string result{reinterpret_cast<const char*>(value)};
    xmlFree(value);
    return result;
}

void attributes(const xmlNode* node, std::initializer_list<std::string_view> supported) {
    if (node->ns != nullptr || node->nsDef != nullptr) invalid(node, "XML namespaces are unsupported.");
    for (auto* attr = node->properties; attr != nullptr; attr = attr->next) {
        const std::string_view name{reinterpret_cast<const char*>(attr->name)};
        if (attr->ns != nullptr || std::ranges::find(supported, name) == supported.end()) {
            invalid(node, "Unsupported attribute '" + std::string(name) + "'.");
        }
    }
}

void children(const xmlNode* node, std::initializer_list<std::string_view> supported) {
    for (auto* child = node->children; child != nullptr; child = child->next) {
        if (child->type == XML_COMMENT_NODE || xmlIsBlankNode(child)) continue;
        if (child->type != XML_ELEMENT_NODE ||
            std::ranges::find(supported, tag(child)) == supported.end()) {
            invalid(child, "Unsupported content inside <" + std::string(tag(node)) + ">.");
        }
    }
}

const xmlNode* single(const xmlNode* node, std::string_view name) {
    const xmlNode* result = nullptr;
    for (const auto* child = node->children; child != nullptr; child = child->next) {
        if (child->type != XML_ELEMENT_NODE || tag(child) != name) continue;
        if (result != nullptr) invalid(child, "Expected one <" + std::string(name) + ">.");
        result = child;
    }
    return result;
}

std::string required(const xmlNode* node, const char* key) {
    const auto value = property(node, key);
    if (!value || value->empty()) invalid(node, "Specify '" + std::string(key) + "'.");
    return *value;
}

template <std::size_t Count>
std::array<double, Count> numbers(const xmlNode* node, const char* key) {
    std::istringstream input{required(node, key)};
    input.imbue(std::locale::classic());
    std::array<double, Count> result{};
    for (double& value : result) {
        if (!(input >> value) || !std::isfinite(value) ||
            std::abs(value) > std::numeric_limits<float>::max()) {
            invalid(node, "Invalid numeric value for '" + std::string(key) + "'.");
        }
    }
    if (!(input >> std::ws).eof()) invalid(node, "Wrong number of values in '" + std::string(key) + "'.");
    return result;
}

template <std::size_t Count>
std::array<double, Count> numbersOr(const xmlNode* node, const char* key,
                                   std::array<double, Count> fallback) {
    return property(node, key) ? numbers<Count>(node, key) : fallback;
}

std::string numberList(const auto& values) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    const char* separator = "";
    for (const auto value : values) {
        output << separator << value;
        separator = " ";
    }
    return output.str();
}

std::string scalar(double value) {
    return numberList(std::array{value});
}

xmlNode* append(xmlNode* parent, const char* name,
                std::initializer_list<std::pair<const char*, std::string>> properties = {}) {
    auto* node = xmlNewChild(parent, nullptr, BAD_CAST name, nullptr);
    if (!node) throw std::bad_alloc{};
    for (const auto& [key, value] : properties) {
        if (!xmlNewProp(node, BAD_CAST key, BAD_CAST value.c_str())) throw std::bad_alloc{};
    }
    return node;
}

Vec3 difference(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Quat orientation(const xmlNode* node) {
    auto q = numbersOr<4>(node, "quat", {1, 0, 0, 0});
    const double length = std::hypot(std::hypot(q[0], q[1]), std::hypot(q[2], q[3]));
    if (length < 1e-12) invalid(node, "Quaternion has zero length.");
    for (double& v : q) v /= length;
    return q;
}

Vec3 rotate(const Quat& q, const Vec3& v) {
    const auto [w, x, y, z] = q;
    const Vec3 t{2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]),
                 2 * (x * v[1] - y * v[0])};
    return {v[0] + w * t[0] + y * t[2] - z * t[1],
            v[1] + w * t[1] + z * t[0] - x * t[2],
            v[2] + w * t[2] + x * t[1] - y * t[0]};
}

void origin(xmlNode* parent, const Vec3& position, const Quat& q) {
    const auto ex = rotate(q, {1, 0, 0});
    const auto ey = rotate(q, {0, 1, 0});
    const auto ez = rotate(q, {0, 0, 1});
    const double cosinePitch = std::hypot(ex[0], ex[1]);
    const double pitch = std::atan2(-ex[2], cosinePitch);
    const Vec3 rpy = cosinePitch > 1e-12
        ? Vec3{std::atan2(ey[2], ez[2]), pitch, std::atan2(ex[1], ex[0])}
        : Vec3{0, pitch, std::atan2(-ey[0], ey[1])};
    append(parent, "origin", {{"xyz", numberList(position)}, {"rpy", numberList(rpy)}});
}

void inertia(xmlNode* link, const Vec3& center, const Quat& q, double mass,
             const std::array<double, 6>& matrix) {
    auto* dest = append(link, "inertial");
    origin(dest, center, q);
    append(dest, "mass", {{"value", scalar(mass)}});
    append(dest, "inertia", {{"ixx", scalar(matrix[0])}, {"iyy", scalar(matrix[1])},
        {"izz", scalar(matrix[2])}, {"ixy", scalar(matrix[3])}, {"ixz", scalar(matrix[4])},
        {"iyz", scalar(matrix[5])}});
}

using Motors = std::map<std::string, double>;

std::vector<MJCFMotor> readMotors(const xmlNode* section) {
    std::vector<MJCFMotor> motors;
    if (!section) return motors;
    std::set<std::string> joints;
    std::set<std::string> names;
    attributes(section, {});
    children(section, {"motor"});
    for (const auto* node = section->children; node != nullptr; node = node->next) {
        if (node->type != XML_ELEMENT_NODE) continue;
        attributes(node, {"name", "joint", "gear", "ctrlrange", "ctrllimited"});
        children(node, {});
        const auto joint = required(node, "joint");
        const auto range = numbers<2>(node, "ctrlrange");
        const auto mode = property(node, "ctrllimited").value_or("auto");
        const double gear = numbersOr<1>(node, "gear", {1})[0];
        if (range[1] <= 0 || range[0] != -range[1] || gear <= 0 ||
            gear * range[1] > std::numeric_limits<float>::max() ||
            (mode != "auto" && mode != "true")) {
            invalid(node, "Use positive gear and a symmetric, enforced motor ctrlrange.");
        }
        if (!joints.insert(joint).second) {
            invalid(node, "Only one motor per joint is supported.");
        }
        const auto name = property(node, "name");
        if (name && (name->empty() || !names.insert(*name).second)) {
            invalid(node, "Motor names must be nonempty and unique when provided.");
        }
        motors.push_back({name.value_or(""), joint, gear * range[1]});
    }
    return motors;
}

void geometry(const xmlNode* node, xmlNode* link, const Vec3& pivot, std::size_t& materialIndex) {
    attributes(node, {"name", "type", "size", "pos", "quat", "rgba", "contype", "conaffinity"});
    children(node, {});
    const auto type = property(node, "type").value_or("sphere");
    const auto position = difference(numbersOr<3>(node, "pos", {}), pivot);
    const auto q = orientation(node);
    const auto rgba = numbersOr<4>(node, "rgba", {0.5, 0.5, 0.5, 1});
    if (std::ranges::any_of(rgba, [](double c) { return c < 0 || c > 1; })) {
        invalid(node, "Color components must be between zero and one.");
    }

    const double group = numbersOr<1>(node, "contype", {1})[0];
    const double mask = numbersOr<1>(node, "conaffinity", {1})[0];
    if (group != mask || (group != 0 && group != 1)) {
        invalid(node, "Use the default contact masks or set both to zero for a visual.");
    }

    const auto writeShape = [&](xmlNode* parent) {
        origin(parent, position, q);
        auto* shape = append(parent, "geometry");
        if (type == "box") {
            auto size = numbers<3>(node, "size");
            for (auto& x : size) {
                if (x <= 0) invalid(node, "Box half-extents must be positive.");
                x *= 2;
            }
            append(shape, "box", {{"size", numberList(size)}});
        } else if (type == "sphere") {
            const auto [radius] = numbers<1>(node, "size");
            if (radius <= 0) invalid(node, "Sphere radius must be positive.");
            append(shape, "sphere", {{"radius", scalar(radius)}});
        } else if (type == "cylinder") {
            const auto [radius, halfLength] = numbers<2>(node, "size");
            if (radius <= 0 || halfLength <= 0) invalid(node, "Cylinder dimensions must be positive.");
            append(shape, "cylinder", {{"radius", scalar(radius)}, {"length", scalar(2 * halfLength)}});
        } else {
            invalid(node, "Unsupported body geom type '" + type + "'. Use box, sphere, or cylinder.");
        }
    };
    auto* visual = append(link, "visual");
    writeShape(visual);
    auto* material = append(visual, "material", {{"name", "mjcf_color_" + std::to_string(++materialIndex)}});
    append(material, "color", {{"rgba", numberList(rgba)}});
    if (group == 1) writeShape(append(link, "collision"));
}

struct Converter {
    xmlNode* robot;
    Motors motors;
    bool degrees;
    std::set<std::string> bodyNames{"numi_mjcf_world"};
    std::set<std::string> jointNames;
    std::size_t fixedIndex = 0;
    std::size_t materialIndex = 0;

    void body(const xmlNode* node, const std::string& parent, const Vec3& parentPivot) {
        attributes(node, {"name", "pos", "quat"});
        children(node, {"body", "joint", "inertial", "geom"});
        const auto name = required(node, "name");
        if (!bodyNames.insert(name).second) invalid(node, "Duplicate or reserved body name.");
        auto* link = append(robot, "link", {{"name", name}});
        const auto* sourceJoint = single(node, "joint");
        Vec3 pivot{};
        std::string jointName = "numi_mjcf_fixed_" + std::to_string(++fixedIndex);
        std::string jointType = "fixed";
        std::optional<std::array<double, 2>> range;
        if (sourceJoint) {
            attributes(sourceJoint, {"name", "type", "pos", "axis", "range", "limited", "damping"});
            children(sourceJoint, {});
            jointName = required(sourceJoint, "name");
            if (jointName.starts_with("numi_mjcf_fixed_")) invalid(sourceJoint, "Reserved joint name.");
            const auto type = property(sourceJoint, "type").value_or("hinge");
            const auto limited = property(sourceJoint, "limited").value_or("auto");
            if (limited != "auto" && limited != "true" && limited != "false") {
                invalid(sourceJoint, "Invalid 'limited' value.");
            }
            if (property(sourceJoint, "range")) {
                const auto authored = numbers<2>(sourceJoint, "range");
                if (limited != "false") range = authored;
            }
            if (range) {
                if (type == "hinge" && degrees) {
                    for (auto& end : *range) end *= std::numbers::pi / 180;
                }
                if ((*range)[0] >= (*range)[1]) invalid(sourceJoint, "Range must increase.");
            } else if (limited == "true") {
                invalid(sourceJoint, "Specify an increasing joint range.");
            }
            if (type == "hinge") jointType = range ? "revolute" : "continuous";
            else if (type == "slide" && range) jointType = "prismatic";
            else invalid(sourceJoint, "Use a hinge or a limited slide joint.");
            pivot = numbersOr<3>(sourceJoint, "pos", {});
        }

        if (!jointNames.insert(jointName).second) invalid(node, "Duplicate joint name '" + jointName + "'.");
        auto* joint = append(robot, "joint", {{"name", jointName}, {"type", jointType}});
        append(joint, "parent", {{"link", parent}});
        append(joint, "child", {{"link", name}});
        auto position = difference(numbersOr<3>(node, "pos", {}), parentPivot);
        const auto q = orientation(node);
        const auto rotatedPivot = rotate(q, pivot);
        for (std::size_t i = 0; i < position.size(); ++i) position[i] += rotatedPivot[i];
        origin(joint, position, q);

        if (sourceJoint) {
            append(joint, "axis", {{"xyz", numberList(numbersOr<3>(sourceJoint, "axis", {0, 0, 1}))}});
            const double damping = numbersOr<1>(sourceJoint, "damping", {0})[0];
            if (damping < 0) invalid(sourceJoint, "Damping must be nonnegative.");
            append(joint, "dynamics", {{"damping", scalar(damping)}});
            const auto motor = motors.find(jointName);
            const auto [lower, upper] = range.value_or(std::array{0.0, 0.0});
            append(joint, "limit", {{"lower", scalar(lower)}, {"upper", scalar(upper)},
                {"effort", motor == motors.end() ? "0" : scalar(motor->second)}});
            if (motor != motors.end()) {
                auto* transmission = append(robot, "transmission", {{"name", jointName}});
                append(transmission, "joint", {{"name", jointName}});
                motors.erase(motor);
            }
        }

        const auto* sourceInertia = single(node, "inertial");
        if (!sourceInertia) invalid(node, "Specify mass and inertia explicitly for every body.");
        attributes(sourceInertia, {"pos", "quat", "mass", "diaginertia", "fullinertia"});
        children(sourceInertia, {});
        const bool full = property(sourceInertia, "fullinertia").has_value();
        if (full && (property(sourceInertia, "diaginertia") || property(sourceInertia, "quat"))) {
            invalid(sourceInertia, "Full inertia is expressed directly in the body frame.");
        }
        auto matrix = full ? numbers<6>(sourceInertia, "fullinertia") : std::array<double, 6>{};
        if (!full) {
            const auto diagonal = numbers<3>(sourceInertia, "diaginertia");
            std::ranges::copy(diagonal, matrix.begin());
        }
        inertia(link, difference(numbers<3>(sourceInertia, "pos"), pivot),
                orientation(sourceInertia), numbers<1>(sourceInertia, "mass")[0], matrix);

        for (const auto* child = node->children; child != nullptr; child = child->next) {
            if (child->type != XML_ELEMENT_NODE) continue;
            if (tag(child) == "geom") geometry(child, link, pivot, materialIndex);
            else if (tag(child) == "body") body(child, name, pivot);
        }
    }
};

} // namespace

MJCFImport importMJCF(const std::filesystem::path& path) {
    MJCFImport result;
    decltype(result.urdf) source{xmlReadFile(path.c_str(), nullptr,
        XML_PARSE_NONET | XML_PARSE_NOERROR | XML_PARSE_NOWARNING), xmlFreeDoc};
    auto* root = source ? xmlDocGetRootElement(source.get()) : nullptr;
    if (!root || source->intSubset || source->extSubset || tag(root) != "mujoco") {
        throw std::runtime_error("Expected a standalone MJCF <mujoco> document without an XML document type.");
    }
    attributes(root, {"model"});
    children(root, {"compiler", "option", "worldbody", "actuator"});
    bool degrees = true;
    if (const auto* compiler = single(root, "compiler")) {
        attributes(compiler, {"angle"});
        children(compiler, {});
        const auto angle = property(compiler, "angle").value_or("degree");
        if (angle != "degree" && angle != "radian") invalid(compiler, "Use degree or radian angles.");
        degrees = angle == "degree";
    }
    if (const auto* option = single(root, "option")) {
        attributes(option, {"gravity", "timestep"});
        children(option, {});
        const auto [x, y, z] = numbersOr<3>(option, "gravity", {0, 0, -9.81});
        if (x != 0 || y != 0 || z > 0 || z < -100) invalid(option, "Use downward Z gravity from 0 to 100 m/s².");
        result.gravity = static_cast<float>(-z);
        const double timestep = numbersOr<1>(option, "timestep", {0.002})[0];
        const double count = timestep > 0 ? 0.02 / timestep : 0;
        const double rounded = std::round(count);
        if (rounded < 1 || rounded > 100 || std::abs(count - rounded) > 1e-8) {
            invalid(option, "Timestep must divide the 20 ms control period into 1–100 equal substeps.");
        }
        result.physicsSubsteps = static_cast<unsigned>(rounded);
    }
    const auto* world = single(root, "worldbody");
    if (!world) invalid(root, "Specify a worldbody.");
    attributes(world, {});
    children(world, {"body", "geom"});
    result.motors = readMotors(single(root, "actuator"));
    if (result.motors.empty()) invalid(root, "Specify at least one direct joint motor.");
    result.urdf.reset(xmlNewDoc(BAD_CAST "1.0"));
    auto* robot = xmlNewNode(nullptr, BAD_CAST "robot");
    if (!result.urdf || !robot) {
        if (robot) xmlFreeNode(robot);
        throw std::bad_alloc{};
    }
    xmlDocSetRootElement(result.urdf.get(), robot);
    const auto name = property(root, "model").value_or("Imported MJCF");
    if (!xmlNewProp(robot, BAD_CAST "name", BAD_CAST name.c_str())) throw std::bad_alloc{};
    auto* link = append(robot, "link", {{"name", "numi_mjcf_world"}});
    inertia(link, {}, {1, 0, 0, 0}, 1, {1, 1, 1, 0, 0, 0});
    Converter converter{robot, {}, degrees};
    for (const auto& motor : result.motors) {
        converter.motors.emplace(motor.joint, motor.effortLimit);
    }
    for (const auto* node = world->children; node != nullptr; node = node->next) {
        if (node->type != XML_ELEMENT_NODE) continue;
        if (tag(node) == "body") {
            converter.body(node, "numi_mjcf_world", {});
            continue;
        }
        attributes(node, {"type", "name", "size", "pos"});
        children(node, {});
        const auto size = numbers<3>(node, "size");
        if (result.ground || property(node, "type") != "plane" ||
            numbersOr<3>(node, "pos", {}) != Vec3{} || size[0] < 0 ||
            size[1] < 0 || size[2] <= 0) {
            invalid(node, "Use at most one horizontal world plane at Z = 0.");
        }
        result.ground = true;
    }
    if (!converter.motors.empty()) {
        invalid(root, "Motor references an unknown joint '" + converter.motors.begin()->first + "'.");
    }
    return result;
}

} // namespace numi
