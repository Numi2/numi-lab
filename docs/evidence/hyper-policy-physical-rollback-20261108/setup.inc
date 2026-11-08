    RunManifest m;
    m.id = "hyper_capacity_probe";
    m.robot = *builtinRobotPack("franka_panda");
    m.scene = makeFrankaPickPlaceScenePack();
    m.sensors.id = "probe_sensors";
    m.task =
        makeFrankaPickPlaceTaskPack(m.sensors.observation, m.reality.reset);
    m.reality.id = "nominal";
    m.profile.id = "probe";
    m.profile.environmentCount = 2;
    m.profile.controlSteps = 30;
    m.profile.physicsSubsteps = 4;
    m.profile.controlTimestepSeconds = .02f;
    m.profile.capacities.rawContacts = 1;
    m.teacher.id = "none";
    CompiledRun run;
    req(compileRun(m, run), "compile run");
    auto const &t = run.task();
    auto const &l = t.layout();
    std::cerr << "frame " << l.actorObservationSize << " actions "
              << l.actionCount << " contacts " << t.contactGroups().size()
              << "\n";
    require(t.contactGroups().size() > 0, "task needs contact group");
    HyperPolicyPack p;
    p.id = "capacity_diagnostic";
    p.revision = 1;
    p.sourceMotionFingerprint = 1;
    p.contract = {.version = 1,
                  .worldFingerprint = t.worldFingerprint(),
                  .taskFingerprint = t.fingerprint(),
                  .observationFingerprint = t.observationFingerprint(),
                  .actionFingerprint = t.actionFingerprint()};
    p.observationMean.assign(l.actorObservationSize, 0);
    p.observationInverseStandardDeviation.assign(l.actorObservationSize, 1);
    p.actionBias.assign(l.actionCount, 0);
    p.actionScale.assign(l.actionCount, 1);
    p.actionLower.assign(l.actionCount, -1);
    p.actionUpper.assign(l.actionCount, 1);
    p.maximumActionRate.assign(l.actionCount, .25f);
    HyperPolicyLayer layer;
    layer.inputCount = l.actorObservationSize;
    layer.outputCount = l.actionCount;
    layer.rank = 1;
    layer.activation = PolicyActivation::identity;
    layer.weights.assign(layer.inputCount * layer.outputCount, 0);
    layer.bias.assign(l.actionCount, 1);
    layer.adapterDown.assign(layer.inputCount, 0);
    layer.adapterUp.assign(layer.outputCount, 0);
    layer.adapterBiasBasis.assign(layer.outputCount, 0);
    p.layers = {layer};
    p.coefficientLimits = {1};
    p.knotPhases = {0, 1};
    p.coefficientKnots = {0, 0};
    p.coefficientTangents = {0, 0};
    p.authorityKnots.assign(2 * l.actionCount, 1);
    p.authorityTangents.assign(2 * l.actionCount, 0);
    p.phaseRateKnots = {1, 1};
    p.phaseRateTangents = {0, 0};
    p.referencePhases = {0, 1};
    p.referenceActions.assign(2 * l.actionCount, 0);
    p.signatureCount = 2 * l.actionCount + 10;
    p.referenceSignatures.assign(2 * p.signatureCount, 0);
    p.signatureWeights.assign(p.signatureCount, 1);
    p.contactTrackCount = 1;
    p.referenceContactMasks = {0, 0};
    p.contactGroupIndices = {0};
    p.phaseAlignmentBlend = 0;
    p.controlTimeStep = .02f;
    CompiledHyperPolicyProgram cp;
    auto hp = compileHyperPolicyProgram(p, t, cp);
    if (!hp.succeeded())
      throw std::runtime_error("compile policy " + hp.element + ": " +
                               hp.message);
