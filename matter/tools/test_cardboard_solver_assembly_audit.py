#!/usr/bin/env python3
"""Failure injection for bulk force, impulse sign, timestep and gather audit."""
import copy
import unittest
import numpy as np
import cardboard_solver_assembly_audit as audit


def fixture(dt=.1):
    header=dict(environment_count=1,node_count_per_environment=4,tetrahedron_count_per_environment=1,
        dt_seconds_by_environment=[dt],epsilon_values=[.03125,.015625],
        materials=[dict(index=2,name='starch2007_finite_glue_elastic',parameter_offset=0,
                        parameter_names=['density','mu','lambda'])],parameters_by_environment=[[1.,2.,3.]])
    h=np.diag([.1,.2,.3]); tangent=np.diag([2.2,2.6,3.0]); force=audit.force(tangent,np.eye(3),1/6)
    dv=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.2,0.],[0.,0.,.3]])
    masses=np.arange(1,5,dtype=float)
    t=dict(environment=0,tet_id=0,material_index=2,node_ids=[0,1,2,3],frame_xyzw=[0,0,0,1],
        inverse_rest_rows=np.eye(3).tolist(),reference_volume_m3=1/6,
        center=dict(F_local=np.eye(3).tolist(),local_F_direction_per_unit_velocity=h.tolist(),
                    P_local_Pa=np.zeros((3,3)).tolist(),P_world_Pa=np.zeros((3,3)).tolist(),
                    dP_local_Pa_per_mps=tangent.tolist(),dP_world_Pa_per_mps=tangent.tolist(),
                    element_force_node_world_N=np.zeros((4,3)).tolist(),
                    operator_element_force_direction_nodes_N_per_s=force.tolist()),perturbations=[])
    nodes=[]
    for i in range(4):
        n=dict(environment=0,node_id=i,accepted=dict(mass_kg=masses[i],v_m_s=[0.,0.,0.]),
               center=dict(mass_kg=masses[i],v_m_s=[0.,0.,0.],constraint_tag=0),
               correction_delta_v_m_s=dv[i].tolist(),
               production_A_delta_v_impulse_Ns=(masses[i]*dv[i]-dt*dt*force[i]).tolist()+[0.],
               center_residual_impulse_Ns=[0.,0.,0.,0.],epsilon_pair_residual_impulses_Ns=[],
               epsilon_pair_realized_u_delta_m=[],epsilon_pair_realized_velocity_delta_m_s=[])
        nodes.append(n)
    for ep,epsilon in enumerate(header['epsilon_values']):
        for sign in (1,-1):
            f=np.eye(3)+sign*epsilon*dt*h
            inv=np.linalg.inv(f)
            p=2*f+(3*np.log(np.linalg.det(f))-2)*inv.T
            t['perturbations'].append(dict(P_world_Pa=p.tolist()))
            side_force=audit.force(p,np.eye(3),1/6)
            key=f"epsilon_pair_{ep}_{'plus' if sign==1 else 'minus'}"
            for i,n in enumerate(nodes):
                v=sign*epsilon*dv[i]
                n[key]=dict(v_m_s=v.tolist())
                n['epsilon_pair_residual_impulses_Ns'].append((dt*side_force[i]-masses[i]*v).tolist()+[0.])
                n['epsilon_pair_realized_u_delta_m'].append((dt*v).tolist())
                n['epsilon_pair_realized_velocity_delta_m_s'].append(v.tolist())
    return header,nodes,[t]


class AssemblyAuditTests(unittest.TestCase):
    def test_mass_stiffness_sign_and_dt_are_reconstructed(self):
        for dt in (.1,.025):
            h,n,t=fixture(dt);r,a=audit.audit_environment(h,n,t,{},0)
            self.assertLess(r['production_gather']['absolute_error'],1e-15)
            self.assertLess(r['captured_candidate_FP64_assembled_action']['absolute_error'],1e-15)
            self.assertGreater(r['mass_action_l2_Ns'],0)
            self.assertGreater(r['captured_stiffness_action_l2_Ns'],0)
            self.assertEqual(r['strict_gate'],'inconclusive_no_measured_noise_floor')
            for side in r['perturbations']:
                self.assertLess(side['residual_gather']['absolute_error'],1e-15)
                self.assertEqual(side['realized_u_velocity_relation']['absolute_error'],0)

    def test_wrong_residual_sign_is_detected(self):
        h,n,t=fixture()
        for node in n:
            node['production_A_delta_v_impulse_Ns']=[-x for x in node['production_A_delta_v_impulse_Ns']]
        r,_=audit.audit_environment(h,n,t,{},0)
        self.assertGreater(r['production_gather']['relative_error'],1)

    def test_wrong_element_node_gather_is_detected(self):
        h,n,t=fixture(.5);t[0]['node_ids']=[1,0,2,3]
        r,_=audit.audit_environment(h,n,t,{},0)
        self.assertGreater(r['production_gather']['relative_error'],.01)

    def test_element_frame_error_is_separate_from_gather(self):
        h,n,t=fixture();t[0]['center']['dP_world_Pa_per_mps'][0][0]+=1
        r,a=audit.audit_environment(h,n,t,{},0)
        self.assertGreater(a[0]['tangent_frame']['relative_error'],.1)
        self.assertEqual(r['production_gather']['absolute_error'],0)

    def test_missing_or_duplicate_nodes_are_rejected(self):
        h,n,t=fixture()
        for broken in (n[:-1],n+[copy.deepcopy(n[0])]):
            with self.assertRaisesRegex(ValueError,'node identity'):
                audit.audit_environment(h,broken,t,{},0)

    def test_mass_mutation_is_rejected(self):
        h,n,t=fixture();n[0]['center']['mass_kg']+=1
        with self.assertRaisesRegex(ValueError,'masses differ'):
            audit.audit_environment(h,n,t,{},0)

    def test_invalid_timestep_is_rejected(self):
        h,n,t=fixture();h['dt_seconds_by_environment']=[0]
        with self.assertRaisesRegex(ValueError,'timestep'):
            audit.audit_environment(h,n,t,{},0)

    def test_unsupported_material_is_rejected(self):
        h,n,t=fixture();h['materials'][0]['name']='unknown'
        with self.assertRaisesRegex(ValueError,'unsupported'):
            audit.audit_environment(h,n,t,{},0)

    def test_zero_reference_never_passes(self):
        r=audit.metric([0.,0.],[0.,0.])
        self.assertIsNone(r['relative_error'])
        self.assertFalse(r['nonzero_reference'])
        self.assertEqual(r['verdict'],'descriptive_only_no_measured_noise_floor')


if __name__=='__main__': unittest.main()
