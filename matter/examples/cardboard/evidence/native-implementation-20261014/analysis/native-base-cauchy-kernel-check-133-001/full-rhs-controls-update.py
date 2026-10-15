from pathlib import Path
B=Path('/Users/home/cardboard-implementation-20261014');D=B/'analysis/native-base-cauchy-kernel-check-133-001';P=D/'native_cauchy_kernel_check.mm';s=P.read_text();(D/'checker-before-full-rhs-controls.mm').write_text(s)
s=s.replace('''    const auto rhsB=makeBuffer(device.gpu,p.rhs);
    const auto baseB=makeBuffer(device.gpu,std::vector<nm_float4>(count));''','''    auto fullRHS=p.rhs;fullRHS.resize(2u*count);
    NMFGMRESLayoutGPU layout{};layout.unknownCount=static_cast<std::uint32_t>(2u*count);
    const auto rhsB=makeBuffer(device.gpu,fullRHS);
    const auto baseB=makeBuffer(device.gpu,std::vector<nm_float4>(2u*count));''')
s=s.replace('invoke(@"numi_matter_metal::nm_fgmres_capture_base_cauchy_rhs",count,','invoke(@"numi_matter_metal::nm_fgmres_capture_base_cauchy_rhs",2u*count,').replace('''[enc setBuffer:baseB offset:0u atIndex:3u];
    });save''','''[enc setBuffer:baseB offset:0u atIndex:3u];[enc setBytes:&layout length:sizeof(layout) atIndex:30u];
    });save''')
s=s.replace('''    NMFGMRESLayoutGPU layout{};layout.unknownCount=static_cast<std::uint32_t>(2u*count);
    const std::uint32_t iteration''','''    const std::uint32_t iteration''')
s=s.replace('''    invoke(@"numi_matter_metal::nm_fgmres_build_base_cauchy_direction",E,''','''    auto build=[&](){invoke(@"numi_matter_metal::nm_fgmres_build_base_cauchy_direction",E,''')
s=s.replace('''[enc setBytes:&iteration length:sizeof(iteration) atIndex:13u];[enc setBytes:&layout length:sizeof(layout) atIndex:30u];
    });''','''[enc setBytes:&iteration length:sizeof(iteration) atIndex:13u];[enc setBuffer:baseB offset:0u atIndex:14u];[enc setBytes:&layout length:sizeof(layout) atIndex:30u];
    });};build();''')
s=s.replace('''need(std::memcmp(rhsB.contents,p.rhs.data(),rhsB.length)==0''','''need(std::memcmp(rhsB.contents,fullRHS.data(),rhsB.length)==0''')
needle='''    std::ofstream report(output/"native-contracts.json");'''
insert='''    // Restore an otherwise healthy selected environment, then inject only
    // a reserved field-half equation. It must reject rather than discard it.
    std::memcpy(metricsB.contents,metrics.data(),metricsB.length);
    std::memcpy(statesB.contents,states.data(),statesB.length);
    std::memcpy(solutionB.contents,solution.data(),solutionB.length);
    std::memset(statusB.contents,0,statusB.length);
    static_cast<nm_float4*>(baseB.contents)[count].x=1.0e-9f;
    build();save(output/"nonzero-field-status.bin",statusB);save(output/"nonzero-field-metrics.bin",metricsB);
    need(static_cast<const NMMatterStatusGPU*>(statusB.contents)[0].code!=0u,"nonzero field RHS silently dropped");
    need(static_cast<const NMMatterStatusGPU*>(statusB.contents)[1].code==0u,"field rejection leaked into unselected env");
    need(std::memcmp(solutionB.contents,solution.data(),solutionB.length)==0,"failed field direction overwrote solution");
    // A nonfinite free mechanical base row is independently rejected even
    // when the already supplied g and Ag happen to be finite.
    std::memcpy(metricsB.contents,metrics.data(),metricsB.length);
    std::memcpy(statesB.contents,states.data(),statesB.length);
    std::memset(statusB.contents,0,statusB.length);
    static_cast<nm_float4*>(baseB.contents)[count].x=0.0f;
    auto freeNode=std::find_if(p.candidate.begin(),p.candidate.begin()+N,[](const auto& n){return n.restAndFixed.w==0.0f&&n.positionAndMass.w>0.0f;});
    need(freeNode!=p.candidate.begin()+N,"fixture has no free node");
    static_cast<nm_float4*>(baseB.contents)[static_cast<std::size_t>(freeNode-p.candidate.begin())].x=std::numeric_limits<float>::quiet_NaN();
    build();save(output/"nonfinite-base-status.bin",statusB);
    need(static_cast<const NMMatterStatusGPU*>(statusB.contents)[0].code!=0u,"nonfinite free base RHS silently dropped");
    need(static_cast<const NMMatterStatusGPU*>(statusB.contents)[1].code==0u,"nonfinite rejection leaked into unselected env");
    need(std::memcmp(solutionB.contents,solution.data(),solutionB.length)==0,"invalid direction overwrote solution");
'''
assert needle in s;s=s.replace(needle,insert+needle);P.write_text(s);(D/'full-rhs-controls-update.py').write_bytes(Path(__file__).read_bytes())
