from pathlib import Path
root=Path(__file__).resolve().parent
s=(root.parents[2]/"tests/hyper_policy_capacity_probe.mm").read_text()
s=s[:s.index("    std::cout << std::setprecision(10) << \"{\\\"runs\\\":[\";")]
s=s.replace("  bool reject;\n};","  bool reject;\n  __strong id<MTLBuffer> policyReset;\n};")
s=s.replace("  if (!c->inner.encode(c->inner.context, p))", """  auto adjusted=p;
  if (c->policyReset != nil) {
    adjusted.resetMasks=(__bridge void*)c->policyReset;
    adjusted.resetMaskOffsetElements=0;
  }
  if (!c->inner.encode(c->inner.context, adjusted))""")
s += r"""
    std::cout << std::setprecision(10) << "{\"runs\":[";
    bool firstOutput=true;
    for (unsigned initial : {1u,2u}) for (unsigned abort : {0u,1u}) for (unsigned repeat=0;repeat<2;++repeat) {
      MetalHyperPolicyRuntime policy;
      MetalHyperPolicyConfiguration pc;pc.metallibPath=argv[1];
      req(createMetalHyperPolicyRuntime(cp,t,pc,policy),"create policy");
      MetalWorldContext world;
      MetalWorldStepConfig cfg;cfg.timestepSeconds=.02f;cfg.physicsSubsteps=4;
      cfg.actuationMode=MetalWorldActuationMode::implicitPositionDrive;
      cfg.taskProgram=t;cfg.ccdMode=MetalWorldCCDMode::disabled;
      auto inner=policy.actionProgram();
      id<MTLDevice> device=MTLCreateSystemDefaultDevice();
      Capture cap{inner,nil,nil,false,nil};
      cfg.deviceActionProgram=inner;cfg.deviceActionProgram.context=&cap;
      cfg.deviceActionProgram.encode=&capturePolicy;
      cfg.deviceActionProgram.afterStep=&captureAfterStep;
      std::vector<float> firstPhase,secondPhase,firstAction,secondAction;
      for (unsigned stage=0;stage<2;++stage) {
        unsigned env=stage==0?initial:2;
        std::vector<float> q,v;
        std::vector<MRBodyStateGPU> scene;
        for(unsigned i=0;i<env;++i) {
          q.insert(q.end(),run.model().defaultQ.begin(),run.model().defaultQ.end());
          v.insert(v.end(),run.world().nv(),0.f);
          auto bodies=run.defaultSceneBodies();
          scene.insert(scene.end(),bodies.begin(),bodies.end());
          scene[i*4].position.z+=2;scene[i*4+3].position.z+=2;
        }
        std::vector<unsigned> reset(env,0u);
        MetalWorldBatch b{.environmentCount=env,.controlStepCount=1,.initialQ=q,.initialV=v,
                          .policyRevision=1,.resetMasks=reset,.initialSceneBodies=scene};
        cap.actions=[device newBufferWithLength:env*l.actionCount*sizeof(float) options:MTLResourceStorageModeShared];
        cap.resets=[device newBufferWithLength:env*sizeof(unsigned) options:MTLResourceStorageModeShared];
        require(cap.actions && cap.resets,"capture");
        if(stage==1) {
          unsigned policyMask[2]={0u,1u};
          cap.policyReset=[device newBufferWithBytes:policyMask length:sizeof(policyMask) options:MTLResourceStorageModeShared];
          require(cap.policyReset != nil,"policy reset buffer");
        }
        if(stage==1 && abort) {
          cap.reject=true;
          MetalWorldResidentState discarded;MetalWorldSubmission ticket;
          auto result=world.initializeResidentState(run.world(),b,cfg,discarded,ticket);
          require(!result.succeeded() && !ticket.valid(),"expected uncommitted reject");
          cap.reject=false;
        }
        MetalWorldResidentState resident;MetalWorldSubmission ticket;
        auto result=world.initializeResidentState(run.world(),b,cfg,resident,ticket);
        req(result,"submit");
        MetalWorldResult output;result=ticket.wait(output);req(result,"wait");
        require(result.published && result.failedStepCount==0 && result.successfulStepCount==env,"physical status");
        std::vector<float> ph,ac,latent,lp;
        require(policy.copyRolloutTrace(1,env,ph,ac,latent,lp),"trace");
        auto data=static_cast<float*>(cap.actions.contents);
        if(stage==0) { firstPhase=ph;firstAction.assign(data,data+env*l.actionCount); }
        else { secondPhase=ph;secondAction.assign(data,data+env*l.actionCount); }
      }
      if(!firstOutput)std::cout << ",";firstOutput=false;
      std::cout << "{\"initial\":" << initial << ",\"abort\":" << abort << ",\"repeat\":" << repeat
                << ",\"first_phase\":" << firstPhase[0] << ",\"first_action\":" << firstAction[0]
                << ",\"phase0\":" << secondPhase[0] << ",\"phase1\":" << secondPhase[1]
                << ",\"action0\":" << secondAction[0] << ",\"action1\":" << secondAction[l.actionCount] << "}";
    }
    std::cout << "]}\n";return 0;
  } catch(const std::exception& e) {std::cerr << "failure: " << e.what() << "\n";return 2;}
}
"""
# The raw suffix uses C++ escapes, not Python string escapes.
s=s.replace(chr(92)+chr(92)+chr(34),chr(92)+chr(34)).replace(chr(92)+chr(92)+"n",chr(92)+"n")
(root/"probe.mm").write_text(s)
