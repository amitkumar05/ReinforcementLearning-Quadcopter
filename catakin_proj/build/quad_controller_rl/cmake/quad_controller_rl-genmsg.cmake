# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "quad_controller_rl: 0 messages, 1 services")

set(MSG_I_FLAGS "-Igeometry_msgs:/opt/ros/kinetic/share/geometry_msgs/cmake/../msg;-Istd_msgs:/opt/ros/kinetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(quad_controller_rl_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/mnt/hgfs/catakin_proj/src/quad_controller_rl/srv/SetPose.srv" NAME_WE)
add_custom_target(_quad_controller_rl_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "quad_controller_rl" "/mnt/hgfs/catakin_proj/src/quad_controller_rl/srv/SetPose.srv" "geometry_msgs/Quaternion:geometry_msgs/Pose:geometry_msgs/Point"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(quad_controller_rl
  "/mnt/hgfs/catakin_proj/src/quad_controller_rl/srv/SetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/quad_controller_rl
)

### Generating Module File
_generate_module_cpp(quad_controller_rl
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/quad_controller_rl
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(quad_controller_rl_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(quad_controller_rl_generate_messages quad_controller_rl_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/mnt/hgfs/catakin_proj/src/quad_controller_rl/srv/SetPose.srv" NAME_WE)
add_dependencies(quad_controller_rl_generate_messages_cpp _quad_controller_rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(quad_controller_rl_gencpp)
add_dependencies(quad_controller_rl_gencpp quad_controller_rl_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS quad_controller_rl_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(quad_controller_rl
  "/mnt/hgfs/catakin_proj/src/quad_controller_rl/srv/SetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/quad_controller_rl
)

### Generating Module File
_generate_module_eus(quad_controller_rl
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/quad_controller_rl
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(quad_controller_rl_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(quad_controller_rl_generate_messages quad_controller_rl_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/mnt/hgfs/catakin_proj/src/quad_controller_rl/srv/SetPose.srv" NAME_WE)
add_dependencies(quad_controller_rl_generate_messages_eus _quad_controller_rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(quad_controller_rl_geneus)
add_dependencies(quad_controller_rl_geneus quad_controller_rl_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS quad_controller_rl_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(quad_controller_rl
  "/mnt/hgfs/catakin_proj/src/quad_controller_rl/srv/SetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/quad_controller_rl
)

### Generating Module File
_generate_module_lisp(quad_controller_rl
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/quad_controller_rl
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(quad_controller_rl_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(quad_controller_rl_generate_messages quad_controller_rl_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/mnt/hgfs/catakin_proj/src/quad_controller_rl/srv/SetPose.srv" NAME_WE)
add_dependencies(quad_controller_rl_generate_messages_lisp _quad_controller_rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(quad_controller_rl_genlisp)
add_dependencies(quad_controller_rl_genlisp quad_controller_rl_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS quad_controller_rl_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(quad_controller_rl
  "/mnt/hgfs/catakin_proj/src/quad_controller_rl/srv/SetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/quad_controller_rl
)

### Generating Module File
_generate_module_nodejs(quad_controller_rl
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/quad_controller_rl
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(quad_controller_rl_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(quad_controller_rl_generate_messages quad_controller_rl_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/mnt/hgfs/catakin_proj/src/quad_controller_rl/srv/SetPose.srv" NAME_WE)
add_dependencies(quad_controller_rl_generate_messages_nodejs _quad_controller_rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(quad_controller_rl_gennodejs)
add_dependencies(quad_controller_rl_gennodejs quad_controller_rl_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS quad_controller_rl_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(quad_controller_rl
  "/mnt/hgfs/catakin_proj/src/quad_controller_rl/srv/SetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quad_controller_rl
)

### Generating Module File
_generate_module_py(quad_controller_rl
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quad_controller_rl
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(quad_controller_rl_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(quad_controller_rl_generate_messages quad_controller_rl_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/mnt/hgfs/catakin_proj/src/quad_controller_rl/srv/SetPose.srv" NAME_WE)
add_dependencies(quad_controller_rl_generate_messages_py _quad_controller_rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(quad_controller_rl_genpy)
add_dependencies(quad_controller_rl_genpy quad_controller_rl_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS quad_controller_rl_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/quad_controller_rl)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/quad_controller_rl
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(quad_controller_rl_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET std_srvs_generate_messages_cpp)
  add_dependencies(quad_controller_rl_generate_messages_cpp std_srvs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/quad_controller_rl)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/quad_controller_rl
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(quad_controller_rl_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET std_srvs_generate_messages_eus)
  add_dependencies(quad_controller_rl_generate_messages_eus std_srvs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/quad_controller_rl)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/quad_controller_rl
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(quad_controller_rl_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET std_srvs_generate_messages_lisp)
  add_dependencies(quad_controller_rl_generate_messages_lisp std_srvs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/quad_controller_rl)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/quad_controller_rl
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(quad_controller_rl_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET std_srvs_generate_messages_nodejs)
  add_dependencies(quad_controller_rl_generate_messages_nodejs std_srvs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quad_controller_rl)
  install(CODE "execute_process(COMMAND \"/usr/bin/python\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quad_controller_rl\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quad_controller_rl
    DESTINATION ${genpy_INSTALL_DIR}
    # skip all init files
    PATTERN "__init__.py" EXCLUDE
    PATTERN "__init__.pyc" EXCLUDE
  )
  # install init files which are not in the root folder of the generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quad_controller_rl
    DESTINATION ${genpy_INSTALL_DIR}
    FILES_MATCHING
    REGEX "${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/quad_controller_rl/.+/__init__.pyc?$"
  )
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(quad_controller_rl_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET std_srvs_generate_messages_py)
  add_dependencies(quad_controller_rl_generate_messages_py std_srvs_generate_messages_py)
endif()
