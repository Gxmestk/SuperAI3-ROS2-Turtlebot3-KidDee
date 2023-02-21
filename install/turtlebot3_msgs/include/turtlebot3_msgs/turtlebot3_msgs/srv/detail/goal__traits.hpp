// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from turtlebot3_msgs:srv/Goal.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__TRAITS_HPP_
#define TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "turtlebot3_msgs/srv/detail/goal__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace turtlebot3_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const Goal_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: empty
  {
    out << "empty: ";
    rosidl_generator_traits::value_to_yaml(msg.empty, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Goal_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: empty
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "empty: ";
    rosidl_generator_traits::value_to_yaml(msg.empty, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Goal_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace turtlebot3_msgs

namespace rosidl_generator_traits
{

[[deprecated("use turtlebot3_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const turtlebot3_msgs::srv::Goal_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  turtlebot3_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use turtlebot3_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const turtlebot3_msgs::srv::Goal_Request & msg)
{
  return turtlebot3_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<turtlebot3_msgs::srv::Goal_Request>()
{
  return "turtlebot3_msgs::srv::Goal_Request";
}

template<>
inline const char * name<turtlebot3_msgs::srv::Goal_Request>()
{
  return "turtlebot3_msgs/srv/Goal_Request";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::Goal_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::Goal_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<turtlebot3_msgs::srv::Goal_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace turtlebot3_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const Goal_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: new_goal
  {
    out << "new_goal: ";
    rosidl_generator_traits::value_to_yaml(msg.new_goal, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Goal_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: new_goal
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "new_goal: ";
    rosidl_generator_traits::value_to_yaml(msg.new_goal, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Goal_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace turtlebot3_msgs

namespace rosidl_generator_traits
{

[[deprecated("use turtlebot3_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const turtlebot3_msgs::srv::Goal_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  turtlebot3_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use turtlebot3_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const turtlebot3_msgs::srv::Goal_Response & msg)
{
  return turtlebot3_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<turtlebot3_msgs::srv::Goal_Response>()
{
  return "turtlebot3_msgs::srv::Goal_Response";
}

template<>
inline const char * name<turtlebot3_msgs::srv::Goal_Response>()
{
  return "turtlebot3_msgs/srv/Goal_Response";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::Goal_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::Goal_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<turtlebot3_msgs::srv::Goal_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::srv::Goal>()
{
  return "turtlebot3_msgs::srv::Goal";
}

template<>
inline const char * name<turtlebot3_msgs::srv::Goal>()
{
  return "turtlebot3_msgs/srv/Goal";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::Goal>
  : std::integral_constant<
    bool,
    has_fixed_size<turtlebot3_msgs::srv::Goal_Request>::value &&
    has_fixed_size<turtlebot3_msgs::srv::Goal_Response>::value
  >
{
};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::Goal>
  : std::integral_constant<
    bool,
    has_bounded_size<turtlebot3_msgs::srv::Goal_Request>::value &&
    has_bounded_size<turtlebot3_msgs::srv::Goal_Response>::value
  >
{
};

template<>
struct is_service<turtlebot3_msgs::srv::Goal>
  : std::true_type
{
};

template<>
struct is_service_request<turtlebot3_msgs::srv::Goal_Request>
  : std::true_type
{
};

template<>
struct is_service_response<turtlebot3_msgs::srv::Goal_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // TURTLEBOT3_MSGS__SRV__DETAIL__GOAL__TRAITS_HPP_
