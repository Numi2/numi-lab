# American crow movement references and v15 model gap

Research inventory, 2026-11-11. This inventory distinguishes species-specific observations from related-bird hypotheses. Links refer to the original source; no third-party video is included or represented as having been measured.

## American crow evidence

* Fernández-Juricic, O’Rourke and Pitlik, *Visual coverage and scanning behavior in two corvid species: American crow and Western scrub jay*, Journal of Comparative Physiology A 196, 879–888 (2010), https://doi.org/10.1007/s00359-010-0570-0 ; manuscript https://estebanfj.bio.purdue.edu/papers/JCompPhysA%20196.pdf. Ground observations include walking or hopping; head rotations were measured as scanning events. This supplies evidence that the head moves during ground behavior, but not a gait-cycle trajectory, neck extension, stride frequency, or joint angles. Its head-scanning event rate must not be used as a gait frequency.
* Cornell Macaulay Library, American crow video ML469899, Evan Barrientos, 29 June 2011, Juneau, Alaska, described in the catalogue as adult, perch and walk: https://macaulaylibrary.org/video/469899. American crow ML416478, Tim Barksdale, April 2000, Montana, described as ground foraging: https://macaulaylibrary.org/video/416478. These are potential primary visual references, not yet downloaded, watched, annotated or measured in this work. Respect the archive’s permissions and preserve clip identifiers and provenance. Search catalogue: https://search.macaulaylibrary.org/catalog?mediaType=video&sort=rating_rank_desc&taxonCode=amecro&view=list

## Related birds, not species calibration

* Hayes and Alexander, *The hopping gaits of crows*, Journal of Zoology (1983), https://doi.org/10.1111/j.1469-7998.1983.tb05784.x. The abstract reports walking and hopping in four crow species, with foot timing data; it is not a measured American-crow joint-motion dataset.
* Troje and Frost, *Head-bobbing in pigeons: how stable is the hold phase?*, Journal of Experimental Biology 203, 935–940 (2000), https://www.biomotionlab.ca/Text/TrojeFrost00.pdf. Pigeon head motion includes a world-stabilized hold and forward thrust in some walking conditions. It motivates recording head position in both world and torso frames, not imposing pigeon rhythm on a crow.
* Itahara and Kano, 2024, https://pmc.ncbi.nlm.nih.gov/articles/PMC11007591/. Large-billed crow head rotations in visual tasks are relevant to head control but do not establish American-crow walking kinematics.
* Abourachid et al., *A new way of analysing 3D movements in bird locomotion*, https://pubmed.ncbi.nlm.nih.gov/21982408/. Related-bird three-dimensional trunk, limb and head measurements motivate measuring coupled motion; species and conditions differ.

## Physical v15 audit

In src/core/CrowEmbodiment.cpp, the active-head construction uses a neck yaw link and head pitch joint at the same nominal pivot. These permit turning and nodding but not independent forward extension and head orientation, and they do not model an S-shaped cervical chain. The trunk is a rigid free body: it can translate, pitch and roll, but cannot physically stretch. The apparent outline can additionally change from neck pose and plumage, so a silhouette alone does not establish internal torso deformation. Current link masses, shapes, and control parameters are engineering approximations, not crow-specific measurements. Changes in physical topology or inertial distribution require fresh flight and contact qualification, including any previously frozen flight controller.

## Measurement and experimental protocol

For accessible clips, retain only authorized derivatives and record media ID, author, license/permission, frame rate, view, scaling assumptions, visibility and speed. Annotate bill/head, neck base, shoulder, hip, both feet, ground and foot contact on multiple strides, with confidence per landmark. Estimate head motion in the world and torso frames, head-neck reach, body pitch and height, foot timing and left/right phase; report camera projection and occlusion uncertainty. Reserve individuals and clips not used in tuning for comparison. If a reference cannot support a metric, mark it unknown rather than borrowing another species’ value.

Develop a physically articulated neck in an isolated model, with bounded torques, mass and inertia accounting, and a gait controller conditioned on actual support and head/torso state. Before promotion, test neutral pose and mass properties, head reach with independent orientation, dynamic stability and contacts, then requalify flight and the full takeoff/navigation/landing sequence. Renderer movement should follow physical states; any plumage-only deformation must be identified as visual and must not count as physical validation. No anatomical calibration or locomotion qualification follows from this inventory alone.

## Additional corvid motion-capture references

* Itahara and Kano, *Corvid Tracking Studio: A custom-built motion capture system to track head movements of corvids*, 2022, https://doi.org/10.2502/janip.72.1.1. Infrared motion capture of large-billed crows and calibration of the eyes and bill provide a relevant head-motion measurement method. This is not American-crow stride data.
* Sellers and Hirasaki, *Markerless 3D motion capture for animal locomotion studies*, Biology Open 3, 656–668 (2014), https://doi.org/10.1242/bio.20148086. Their brief free-flight crow example includes 3D head and wingtip paths; the paper reports approximately 4.74 m/s ground speed and 0.82 m/s ascent for that single observed flight, with low temporal resolution for flapping. Species is not established here; these are flight observations, not walking targets or species-wide statistics. The methods may help future multi-view calibration.

## Corvid hindlimb and takeoff data

* Stoessel and Fischer, *Comparative intralimb coordination in avian bipedal locomotion*, J Exp Biol 215:4055–4069 (2012), https://doi.org/10.1242/jeb.070458. High-speed biplanar X-ray quantified limb joint and segment angles and pelvic rotations during treadmill locomotion in the jackdaw, Corvus monedula, and two other bird species. The abstract describes differences in side-to-side limb and pelvic motion, including increased jackdaw hip mobility. The full paper was not accessible in this search, and no numerical joint trajectories were extracted; this is corvid comparative evidence, not American-crow calibration.
* Meilak et al., *On the 3D Nature of the Magpie Functional Hindlimb Anatomy During the Take-Off Jump*, Front Bioeng Biotechnol 9:676894 (2021), https://doi.org/10.3389/fbioe.2021.676894. An estimated 3D magpie hindlimb model used crow histology to inform muscle paths and considered uncertainty. Its model and cross-species motion ranges are useful structural hypotheses; they do not directly measure American-crow takeoff forces or validate current actuator torques.
