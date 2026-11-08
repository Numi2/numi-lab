        metalrobo::RunManifest compatiblePolicySource;
        const metalrobo::RunManifest* compatiblePolicySourcePointer =
            nullptr;
        if (config->interaction_reference_mode ==
            MR_INTERACTION_REFERENCE_GUIDE) {
            manifest.task.interactionControlReference = true;
        } else if (config->interaction_reference_mode ==
                   MR_INTERACTION_REFERENCE_RESET_ONLY) {
            compatiblePolicySource = manifest;
            compatiblePolicySource.task.interactionControlReference = true;
            compatiblePolicySourcePointer = &compatiblePolicySource;
            manifest.task.interactionControlReference = false;
        }
        if (config->override_interaction_student_authority != 0u) {
            manifest.task.interactionStudentAuthority =
                config->interaction_student_authority;
        }
        if (config->override_interaction_reset_phase_fraction != 0u) {
            manifest.task.interactionResetPhaseFraction =
                config->interaction_reset_phase_fraction;
        }
        if (config->override_interaction_reset_phase_probability != 0u) {
            manifest.task.interactionResetPhaseProbability =
                config->interaction_reset_phase_probability;
        }
        if (config->override_interaction_reset_maximum_phase != 0u) {
            manifest.task.interactionResetMaximumPhase =
                config->interaction_reset_maximum_phase;
        }
        manifest.teacher = {
            .id = interactions.id,
            .kind = metalrobo::TeacherKind::motionImagination,
            .provider = "interaction_pack",
            .model = clip.id,
            .artifact = interaction_pack_path,
            .artifactFingerprint = loaded.contentHash,
            .interactions = interactions,
            .interactionClip = clip.id,
        };
