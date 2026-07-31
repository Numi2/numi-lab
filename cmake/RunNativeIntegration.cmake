if(NOT DEFINED METALROBO_SIMULATION OR
   NOT DEFINED METALROBO_TRAIN OR
   NOT DEFINED METALROBO_METALLIB OR
   NOT DEFINED METALROBO_INTEGRATION_DIR)
    message(FATAL_ERROR "Native integration paths are incomplete")
endif()

file(REMOVE_RECURSE "${METALROBO_INTEGRATION_DIR}")
file(MAKE_DIRECTORY "${METALROBO_INTEGRATION_DIR}")

execute_process(
    COMMAND
        "${METALROBO_SIMULATION}"
        --metallib "${METALROBO_METALLIB}"
        --envs 1
        --steps 4
        --repeats 1
        --chunk 1
        --scene ground
        --numi-iteration-policy fixed
        --zero-actions
        --no-scheduled-resets
    RESULT_VARIABLE simulation_status
    OUTPUT_VARIABLE simulation_output
    ERROR_VARIABLE simulation_error
)
if(NOT simulation_status EQUAL 0)
    message(FATAL_ERROR
        "Native simulation integration failed (${simulation_status})\n"
        "${simulation_output}\n${simulation_error}"
    )
endif()

set(initial_policy
    "${METALROBO_INTEGRATION_DIR}/initial.policypack")
set(updated_policy
    "${METALROBO_INTEGRATION_DIR}/updated.policypack")
set(deployment_policy
    "${METALROBO_INTEGRATION_DIR}/deployment.policypack")
set(rollout_pack
    "${METALROBO_INTEGRATION_DIR}/rollout.pack")
set(learner_state
    "${METALROBO_INTEGRATION_DIR}/learner.safetensors")

# Five updates intentionally exceed the three-slot rollout ring. This owns the
# managed MLX lifetime/release contract in addition to the native rollout path.
execute_process(
    COMMAND
        "${METALROBO_TRAIN}"
        --metallib "${METALROBO_METALLIB}"
        --envs 1
        --steps 2
        --updates 5
        --chunk 1
        --scene ground
        --numi-iteration-policy fixed
        --update-epochs 1
        --minibatch-size 1
        --policy-pack "${initial_policy}"
        --initialize-policy native_integration
        --updated-policy-pack "${updated_policy}"
        --deployment-policy-pack "${deployment_policy}"
        --rollout-pack "${rollout_pack}"
        --learner-state "${learner_state}"
    RESULT_VARIABLE training_status
    OUTPUT_VARIABLE training_output
    ERROR_VARIABLE training_error
)
if(NOT training_status EQUAL 0)
    message(FATAL_ERROR
        "Native Swift/MLX integration failed (${training_status})\n"
        "${training_output}\n${training_error}"
    )
endif()

foreach(artifact IN ITEMS
    "${updated_policy}"
    "${deployment_policy}"
    "${rollout_pack}"
    "${learner_state}")
    if(NOT EXISTS "${artifact}")
        message(FATAL_ERROR "Native integration did not publish ${artifact}")
    endif()
    file(SIZE "${artifact}" artifact_size)
    if(artifact_size EQUAL 0)
        message(FATAL_ERROR "Native integration published an empty ${artifact}")
    endif()
endforeach()

string(FIND "${training_output}" "\"final_policy_revision\" : 6"
    final_revision)
string(FIND "${training_output}" "\"failed_environment_steps\" : 0"
    no_failures)
string(FIND "${training_output}" "\"rollout_ring_bytes\""
    ring_evidence)
if(final_revision EQUAL -1 OR
   no_failures EQUAL -1 OR
   ring_evidence EQUAL -1)
    message(FATAL_ERROR
        "Native integration output lacks rollout-ring qualification evidence\n"
        "${training_output}"
    )
endif()

file(REMOVE_RECURSE "${METALROBO_INTEGRATION_DIR}")
