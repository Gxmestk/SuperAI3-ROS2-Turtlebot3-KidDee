// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from turtlebot3_msgs:srv/DrlStep.idl
// generated code does not contain a copyright notice

#ifndef TURTLEBOT3_MSGS__SRV__DETAIL__DRL_STEP__TRAITS_HPP_
#define TURTLEBOT3_MSGS__SRV__DETAIL__DRL_STEP__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "turtlebot3_msgs/srv/detail/drl_step__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace turtlebot3_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const DrlStep_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: action
  {
    if (msg.action.size() == 0) {
      out << "action: []";
    } else {
      out << "action: [";
      size_t pending_items = msg.action.size();
      for (auto item : msg.action) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: previous_action
  {
    if (msg.previous_action.size() == 0) {
      out << "previous_action: []";
    } else {
      out << "previous_action: [";
      size_t pending_items = msg.previous_action.size();
      for (auto item : msg.previous_action) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const DrlStep_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: action
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.action.size() == 0) {
      out << "action: []\n";
    } else {
      out << "action:\n";
      for (auto item : msg.action) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: previous_action
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.previous_action.size() == 0) {
      out << "previous_action: []\n";
    } else {
      out << "previous_action:\n";
      for (auto item : msg.previous_action) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DrlStep_Request & msg, bool use_flow_style = false)
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
  const turtlebot3_msgs::srv::DrlStep_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  turtlebot3_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use turtlebot3_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const turtlebot3_msgs::srv::DrlStep_Request & msg)
{
  return turtlebot3_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<turtlebot3_msgs::srv::DrlStep_Request>()
{
  return "turtlebot3_msgs::srv::DrlStep_Request";
}

template<>
inline const char * name<turtlebot3_msgs::srv::DrlStep_Request>()
{
  return "turtlebot3_msgs/srv/DrlStep_Request";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::DrlStep_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::DrlStep_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<turtlebot3_msgs::srv::DrlStep_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace turtlebot3_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const DrlStep_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: state
  {
    if (msg.state.size() == 0) {
      out << "state: []";
    } else {
      out << "state: [";
      size_t pending_items = msg.state.size();
      for (auto item : msg.state) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: reward
  {
    out << "reward: ";
    rosidl_generator_traits::value_to_yaml(msg.reward, out);
    out << ", ";
  }

  // member: done
  {
    out << "done: ";
    rosidl_generator_traits::value_to_yaml(msg.done, out);
    out << ", ";
  }

  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: distance_traveled
  {
    out << "distance_traveled: ";
    rosidl_generator_traits::value_to_yaml(msg.distance_traveled, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const DrlStep_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.state.size() == 0) {
      out << "state: []\n";
    } else {
      out << "state:\n";
      for (auto item : msg.state) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: reward
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "reward: ";
    rosidl_generator_traits::value_to_yaml(msg.reward, out);
    out << "\n";
  }

  // member: done
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "done: ";
    rosidl_generator_traits::value_to_yaml(msg.done, out);
    out << "\n";
  }

  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: distance_traveled
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "distance_traveled: ";
    rosidl_generator_traits::value_to_yaml(msg.distance_traveled, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DrlStep_Response & msg, bool use_flow_style = false)
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
  const turtlebot3_msgs::srv::DrlStep_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  turtlebot3_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use turtlebot3_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const turtlebot3_msgs::srv::DrlStep_Response & msg)
{
  return turtlebot3_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<turtlebot3_msgs::srv::DrlStep_Response>()
{
  return "turtlebot3_msgs::srv::DrlStep_Response";
}

template<>
inline const char * name<turtlebot3_msgs::srv::DrlStep_Response>()
{
  return "turtlebot3_msgs/srv/DrlStep_Response";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::DrlStep_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::DrlStep_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<turtlebot3_msgs::srv::DrlStep_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<turtlebot3_msgs::srv::DrlStep>()
{
  return "turtlebot3_msgs::srv::DrlStep";
}

template<>
inline const char * name<turtlebot3_msgs::srv::DrlStep>()
{
  return "turtlebot3_msgs/srv/DrlStep";
}

template<>
struct has_fixed_size<turtlebot3_msgs::srv::DrlStep>
  : std::integral_constant<
    bool,
    has_fixed_size<turtlebot3_msgs::srv::DrlStep_Request>::value &&
    has_fixed_size<turtlebot3_msgs::srv::DrlStep_Response>::value
  >
{
};

template<>
struct has_bounded_size<turtlebot3_msgs::srv::DrlStep>
  : std::integral_constant<
    bool,
    has_bounded_size<turtlebot3_msgs::srv::DrlStep_Request>::value &&
    has_bounded_size<turtlebot3_msgs::srv::DrlStep_Response>::value
  >
{
};

template<>
struct is_service<turtlebot3_msgs::srv::DrlStep>
  : std::true_type
{
};

template<>
struct is_service_request<turtlebot3_msgs::srv::DrlStep_Request>
  : std::true_type
{
};

template<>
struct is_service_response<turtlebot3_msgs::srv::DrlStep_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // TURTLEBOT3_MSGS__SRV__DETAIL__DRL_STEP__TRAITS_HPP_
