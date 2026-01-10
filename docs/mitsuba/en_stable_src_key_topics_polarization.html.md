---
url: https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html
crawled_at: 2025-11-13T17:13:40.274444
title: https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Polarization[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#polarization "Link to this heading")
[![../../_images/teaser.jpg](https://mitsuba.readthedocs.io/en/stable/_images/teaser.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/teaser.jpg)
## Introduction[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#introduction "Link to this heading")
The retargetable design of the Mitsuba 3 rendering system can be leveraged to optionally keep track of the full polarization state of light, meaning that it simulates the oscillations of the light’s electromagnetic wave perpendicular to its direction of travel.
Because humans do not perceive this directly, accounting for it is usually not necessary when rendering images that are intended to look realistic. However, polarization is easily observed using a variety of measurement devices and cameras and it tends to provide a wealth of information about the material and shape of visible objects. For this reason, polarization is a powerful tool for solving inverse problems, and this is one of the reasons why we chose to support it in Mitsuba 3.
Polarized rendering has been studied extensively before. A first (unidirectional) algorithm was proposed by Wilkie and Weidlich [[WW12](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id26 "Alexander Wilkie and Andrea Weidlich. Polarised light in computer graphics. In SIGGRAPH Asia 2012 Courses, SA '12. New York, NY, USA, 2012. Association for Computing Machinery. URL: https://doi.org/10.1145/2407783.2407791, doi:10.1145/2407783.2407791.")] before it was extended to bidirectional techniques in two later works [[MSWK16](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id4 "Michal Mojzik, Tomas Skrivan, Alexander Wilkie, and Jaroslav Krivanek. Bi-Directional Polarised Light Transport. In Eurographics Symposium on Rendering. 2016.")], [[JA18](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id3 "Adrian Jarabo and Victor Arellano. Bidirectional rendering of vector light transport. Computer Graphics Forum, 2018.")] by leveraging the general path space formulation [[Vea98](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id25 "Eric Veach. Robust Monte Carlo Methods for Light Transport Simulation. Stanford University, Stanford, CA, USA, 1998. ISBN 0591907801.")].
Polarized rendering is also implemented by the [Advanced Rendering Toolkit (ART) research rendering system](https://cgg.mff.cuni.cz/ART/).
* * *
The first two sections of this document cover the mathematical framework behind polarization that is relevant for rendering (Section [Mathematics of polarized light](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#mathematics-of-polarized-light)) as well as an in-depth description of the _Fresnel equations_ that are important components of many reflectance models (Section [Fresnel equations](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#fresnel-equations)). Finally, Section [Implementation](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#implementation) serves as a _developer guide_ and explains how polarization is implemented in Mitsuba 3.
## Mathematics of polarized light[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#mathematics-of-polarized-light "Link to this heading")
In this section we go over the basics behind the mathematical representation of polarized light and how it interacts with common optical elements. No prior knowledge of polarization is required for following these Sections though we only cover the contents needed for rendering applications. More thorough discussions can be found in the optics literature [[Hec98](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id27 "Eugene Hecht. Optics. Addison-Wesley, 4th edition, 1998.")], [[Col93](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id2 "Edward Collett. Polarized light : fundamentals and application. Marcel Dekker New York, 1993. ISBN 0824787293.")], [[Col05](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id30 "E. Collett. Field Guide to Polarization. Field Guide Series. Society of Photo Optical, 2005. ISBN 9780819458681.")].
Modern rendering systems are generally built on top of the _radiometry_ framework for describing light where algorithms usually track a unit called _radiance_ [[PJH16](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id31 "Matt Pharr, Wenzel Jakob, and Greg Humphreys. Physically Based Rendering: From Theory to Implementation \(3rd ed.\). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 3rd edition, October 2016. ISBN 9780128006450.")]. This was originally based on a particle description of light and thus does not cover the effects of polarization (or other aspects of wave optics).
From the optics community, there are two commonly used frameworks to describe the polarization state of light:
  1. **Mueller-Stokes calculus**


The polarization state (fully polarized, partially polarized, or unpolarized) is represented with a \\(4 \times 1\\) _Stokes vector_ while interaction with optical elements or scattering is achieved by multiplication with \\(4 \times 4\\) _Mueller matrices_.
  1. **Jones calculus**


This representation is simpler and uses \\(2 \times 1\\) and \\(2 \times 2\\) _Jones vectors and matrices_ instead. They are directly related to the underlying electrical field components of the wave and can explain superpositions of waves, e.g. interference effects. However, Jones calculus can only represent fully polarized light and is therefore of limited interest for rendering applications.
Like previous work in polarized rendering [[WW12](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id26 "Alexander Wilkie and Andrea Weidlich. Polarised light in computer graphics. In SIGGRAPH Asia 2012 Courses, SA '12. New York, NY, USA, 2012. Association for Computing Machinery. URL: https://doi.org/10.1145/2407783.2407791, doi:10.1145/2407783.2407791.")], [[MSWK16](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id4 "Michal Mojzik, Tomas Skrivan, Alexander Wilkie, and Jaroslav Krivanek. Bi-Directional Polarised Light Transport. In Eurographics Symposium on Rendering. 2016.")], [[JA18](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id3 "Adrian Jarabo and Victor Arellano. Bidirectional rendering of vector light transport. Computer Graphics Forum, 2018.")], we will use the Mueller-Stokes calculus for our implementation.
### Stokes vectors[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#stokes-vectors "Link to this heading")
#### Definitions[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#definitions "Link to this heading")
[![../../_images/stokes_vector.svg](https://mitsuba.readthedocs.io/en/stable/_images/stokes_vector.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/stokes_vector.svg)
**Figure 1** : Illustration of the individual Stokes vector components \\(\mathbf{s}_0, \mathbf{s}_1, \mathbf{s}_2, \mathbf{s}_3\\) on top of the reference coordinate frame (\\(\mathbf{x}, \mathbf{y}\\)) where the \\(\mathbf{x}\\)-axis is interpreted as horizontal.[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#stokes-components "Link to this image")
A Stokes vector is a 4-dimensional quantity \\(\mathbf{s} = [\mathbf{s}_0, \mathbf{s}_1, \mathbf{s}_2, \mathbf{s}_3]^{\top}\\) [[[1]](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id15)] which parameterizes the full polarization state of light [[[2]](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id16)].
  * \\(\mathbf{s}_0\\) is equivalent to _radiance_ normally used in physically based rendering. It measures the intensity of light but does not say anything about its polarization state.
  * \\(\mathbf{s}_1\\) distinguishes horizontal vs. vertical _linear_ polarization, where \\(\mathbf{s}_1 = \pm 1\\) stands for completely horizontally or vertically polarized light respectively.
  * \\(\mathbf{s}_2\\) is similar to \\(\mathbf{s}_1\\) but distinguishes diagonal linear polarization at \\(\pm 45˚\\) angles, as measured from the horizontal axis.
  * \\(\mathbf{s}_3\\) distinguishes right vs. left circular polarization, where \\(\mathbf{s}_3 = \pm 1\\) stands for full right or left circularly polarized light respectively.
  * Physically plausible Stokes vectors need to fulfill \\(\mathbf{s}_0 \geq \sqrt{\mathbf{s}_1^2 + \mathbf{s}_2^2 + \mathbf{s}_3^2}\\).


A few typical examples of interesting polarization states are summarized in the following table and animations.
Description | Corresponding Stokes vector  
---|---  
Unpolarized light | \\([1, 0, 0, 0]^{\top}\\)  
Horizontal linearly polarized light | \\([1, 1, 0, 0]^{\top}\\)  
Vertical linearly polarized light | \\([1, -1, 0, 0]^{\top}\\)  
Diagonal (+45˚) linearly polarized light | \\([1, 0, 1, 0]^{\top}\\)  
Diagonal (-45˚) linearly polarized light | \\([1, 0, -1, 0]^{\top}\\)  
Right circularly polarized light | \\([1, 0, 0, 1]^{\top}\\)  
Left circularly polarized light | \\([1, 0, 0, -1]^{\top}\\)  
**Table 1** Common Stokes vector values.
#### Reference frames[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#reference-frames "Link to this heading")
As illustrated in [Figure 1](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#stokes-components), these definitions above are only true relative to a given reference coordinate frame (\\(\mathbf{x}, \mathbf{y}\\)). Here, the \\(\mathbf{x}\\)-axis indicates what is meant with “horizontal”. As long as the frame is orthogonal to the beam of light, this is purely a matter of convention and infinitely many such frames exist. We follow the convention used in most textbooks and other sources with a right-handed coordinate system where the z-axis points along the light propagation direction like so [[[3]](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id17)]:
Note how we take the point of view of the “receiver” and look into the direction of the “source” of the beam to describe the Stokes vector. In particular, this also clarifies the handedness of circular polarization.
As a final example, consider [Figure 2](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#stokes-rotation) which shows linearly polarized light in two different reference frames, resulting in two different Stokes vectors.
[![../../_images/stokes_rotation.svg](https://mitsuba.readthedocs.io/en/stable/_images/stokes_rotation.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/stokes_rotation.svg)
**Figure 2** : (**Left**) Linearly polarized light is observed in a reference frame \\((\mathbf{x}, \mathbf{y})\\) where the measured Stokes vector looks like horizontal polarization, \\(\mathbf{s} = [1, 1, 0, 0]^{\top}\\). (**Right**) The same beam is observed in a rotated reference frame \\((\mathbf{x}', \mathbf{y}')\\) where the Stokes vector looks like \\(-45˚\\) linear polarization, \\(\mathbf{s} = [1, 0, -1, 0]^{\top}\\).[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#stokes-rotation "Link to this image")
* * *
### Mueller matrices[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#mueller-matrices "Link to this heading")
Any change of the polarization change due to interaction with some optical element or interface can be summarized as a multiplication of the corresponding Stokes vector with a _Mueller matrix_ \\(\mathbf{M} \in \mathbb{R}^{4x4}\\). After the interaction, the incident (\\(\mathbf{s}_{\text{in}}\\)) and outgoing (\\(\mathbf{s}_{\text{out}}\\)) Stokes vectors are related by
(1)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq1 "Link to this equation")\\[ \mathbf{s}_{\text{out}} = \mathbf{M} \cdot \mathbf{s}_{\text{in}}\\]
Similar to Stokes vectors, Mueller matrices are also only valid in some reference coordinate system. More precisely, every matrix operates from a fixed incident \\((\mathbf{x}_{\text{in}}, \mathbf{y}_{\text{in}})\\) to a fixed outgoing \\((\mathbf{x}_{\text{out}}, \mathbf{y}_{\text{out}})\\) reference frame.
The most common optical elements (such as linear polarizers or retarders, see below) operate along a single direction of a light beam, and in that case, both of their reference frames are usually assumed to be the same, with the \\(\mathbf{x}\\)-axis aligned with the optical table:
[![../../_images/mueller_matrix_frames_aligned_crop.png](https://mitsuba.readthedocs.io/en/stable/_images/mueller_matrix_frames_aligned_crop.png) ](https://mitsuba.readthedocs.io/en/stable/_images/mueller_matrix_frames_aligned_crop.png)
**Figure 3** : Incident (blue) and outgoing (green) Stokes reference frames in the case with collinear directions.[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#frames-collinear "Link to this image")
For the general case, e.g. a Mueller matrix that describes a reflection on some interface, the two frames are necessarily different and need to be tracked carefully:
[![../../_images/mueller_matrix_frames_reflection_crop.png](https://mitsuba.readthedocs.io/en/stable/_images/mueller_matrix_frames_reflection_crop.png) ](https://mitsuba.readthedocs.io/en/stable/_images/mueller_matrix_frames_reflection_crop.png)
**Figure 4** : Incident (blue) and outgoing (green) Stokes reference frames in the general case of reflection.[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#frames-reflection "Link to this image")
Warning
**Stokes and Mueller matrix operations**
A lot of care has to be taken when operating with Stokes vectors and Mueller matrices.
  1. Matrix multiplication between Mueller matrix and Stokes vector \\(\mathbf{s}_{\text{out}} = \mathbf{M} \cdot \mathbf{s}_{\text{in}}\\) like above is only valid if the incident reference frame of \\(\mathbf{M}\\) is equivalent to the reference frame of \\(\mathbf{s}_{\text{in}}\\).
  2. Mueller matrix multiplication \\(\mathbf{M}_2 \cdot \mathbf{M}_1\\) is only valid if the outgoing frame of \\(\mathbf{M}_1\\) is aligned with the incident frame of \\(\mathbf{M}_2\\). Like normal matrix multiplication, this operation does not commute in general.
  3. Mueller matrices need to be _left multiplied_ onto Stokes vectors. For instance, \\(\mathbf{M}_2 \cdot \mathbf{M}_1 \cdot \mathbf{s}_{\text{in}}\\) indicates that a light beam (with Stokes vector \\(\mathbf{s}_{\text{in}}\\)) first interacts with \\(\mathbf{M}_1\\) followed by \\(\mathbf{M}_2\\).


Apart from standard optical elements ([Table 2](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#table-optical-elements)) and other a few other idealized cases, interaction of polarized light with arbitrary materials is not well understood at this point. We will cover the important special case of specular reflection or refraction later in Section [Fresnel equations](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#fresnel-equations).
Description | Mueller matrix  
---|---  
Ideal depolarizer | \\(\begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 0 & 0 & 0 \\\ 0 & 0 & 0 & 0 \\\ 0 & 0 & 0 & 0 \end{bmatrix}\\)  
Attenuation filter, \\(\alpha\\): transmission | \\(\alpha \begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \\\ 0 & 0 & 1 & 0 \\\ 0 & 0 & 0 & 1 \end{bmatrix}\\)  
Ideal linear polarizer (horizontal transmission) | \\(\frac{1}{2} \begin{bmatrix} 1 & 1 & 0 & 0 \\\ 1 & 1 & 0 & 0 \\\ 0 & 0 & 0 & 0 \\\ 0 & 0 & 0 & 0 \end{bmatrix}\\)  
Ideal linear retarder (fast axis horizontal), \\(\phi\\): phase difference | \\(\begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \\\ 0 & 0 & \cos\phi & \sin\phi \\\ 0 & 0 & -\sin\phi & \cos\phi \end{bmatrix}\\)  
Ideal quarter-wave plate (fast axis horizontal) | \\(\begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \\\ 0 & 0 & 0 & 1 \\\ 0 & 0 & -1 & 0 \end{bmatrix}\\)  
Ideal half-wave plate | \\(\begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \\\ 0 & 0 & -1 & 0 \\\ 0 & 0 & 0 & -1 \end{bmatrix}\\)  
Ideal right circular polarizer | \\(\frac{1}{2} \begin{bmatrix} 1 & 0 & 0 & 1 \\\ 0 & 0 & 0 & 0 \\\ 0 & 0 & 0 & 0 \\\ 1 & 0 & 0 & 1 \end{bmatrix}\\)  
Ideal left circular polarizer | \\(\frac{1}{2} \begin{bmatrix} 1 & 0 & 0 & -1 \\\ 0 & 0 & 0 & 0 \\\ 0 & 0 & 0 & 0 \\\ -1 & 0 & 0 & 1 \end{bmatrix}\\)  
General polarizer. \\(\alpha_x, \alpha_y\\): transmission along the two orthogonal axes | \\(\frac{1}{2} \begin{bmatrix} \alpha_x^2 + \alpha_y^2 & \alpha_x^2 - \alpha_y^2 & 0 & 0 \\\ \alpha_x^2 - \alpha_y^2 & \alpha_x^2 + \alpha_y^2 & 0 & 0 \\\ 0 & 0 & 2 \alpha_x \alpha_y & 0 \\\ 0 & 0 & 0 & 2 \alpha_x \alpha_y \end{bmatrix}\\)  
**Table 2** A list of Mueller matrices for typical optical elements.
#### Rotation of Stokes vector frames[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rotation-of-stokes-vector-frames "Link to this heading")
Due to the importance of applying Mueller matrices on Stokes vectors only when expressed in the same reference frames, we often have to perform rotations of these frames during a simulation of polarized light. This is somewhat unintuitive at first and can be a source of common errors in an implementation.
We already briefly touched on how rotations of reference frames change how e.g. a Stokes vector is measured (see [Figure 2](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#stokes-rotation)) and we will now formalize the rotation operation as yet another Mueller matrix \\(\mathbf{R}(\theta)\\) [[[4]](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id20)] that can be applied to a Stokes vector \\(\mathbf{s}\\) to express it in another reference frame that was rotated by an angle \\(\theta\\) (measured counter-clockwise from the \\(\mathbf{x}\\)-axis). This new Stokes vector is then
(2)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq2 "Link to this equation")\\[\begin{equation} \mathbf{s}' = \mathbf{R}(\theta) \cdot \mathbf{s} \end{equation}\\]
with the rotator matrix defined as
(3)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-rotator-matrix "Link to this equation")\\[\begin{split}\begin{equation} \mathbf{R}(\theta) = \begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & \cos(2\theta) & \sin(2\theta) & 0 \\\ 0 & -\sin(2\theta) & \cos(2\theta) & 0 \\\ 0 & 0 & 0 & 1 \end{bmatrix} \end{equation}\end{split}\\]
Here is another visualization of this process. Again, note how the polarization of light did not actually change globally, but only expressed relative to the used reference frames:
#### Rotation of Mueller matrix frames[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rotation-of-mueller-matrix-frames "Link to this heading")
This idea of rotating the Stokes reference frames can naturally be used to address the challenges of Mueller matrix multiplications involving mismatched frames from above. Consider [Figure 5](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rotated-mueller-matrix) below.
We want to apply some Mueller matrix \\(\hat{\mathbf{M}}\\) that operates from reference frame (\\(\hat{\mathbf{x}}_{\text{in}}, \hat{\mathbf{y}}_{\text{in}}\\)) to (\\(\hat{\mathbf{x}}_{\text{out}}, \hat{\mathbf{y}}_{\text{out}}\\)) to an incoming beam of light with Stokes vector \\(\mathbf{s}\\) that is defined relative to the frame (\\(\mathbf{x}_{\text{in}}, \mathbf{y}_{\text{in}}\\)). As the mismatch makes this impossible, we have to first rotate the incident Stokes vector to be expressed in the different frame:
(4)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq4 "Link to this equation")\\[\begin{equation} \mathbf{s}' = \mathbf{R}(\theta_{\text{in}}) \cdot \mathbf{s} \end{equation}\\]
where \\(\theta_{\text{in}}\\) is the relative angle between the two frame bases \\(\mathbf{x}_{\text{in}}\\) and \\(\hat{\mathbf{x}}_{\text{in}}\\).
We can then apply the matrix \\(\hat{\mathbf{M}}\\), as it is valid for the incident frame (\\(\hat{\mathbf{x}}_{\text{in}}, \hat{\mathbf{y}}_{\text{in}}\\)):
(5)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq5 "Link to this equation")\\[\begin{equation} \mathbf{s}'' = \hat{\mathbf{M}} \cdot \mathbf{s}' = \hat{\mathbf{M}} \cdot \mathbf{R}(\theta_{\text{in}}) \cdot \mathbf{s} \end{equation}\\]
The resulting Stokes vector \\(\mathbf{s}''\\) is now valid with respect to the frame (\\(\hat{\mathbf{x}}_{\text{out}}, \hat{\mathbf{y}}_{\text{out}}\\)). As a last step, we might want to, yet again, rotate its frame to be aligned with some other (arbitrary) reference frame (\\(\mathbf{x}_{\text{out}}, \mathbf{y}_{\text{out}}\\)):
(6)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-rotated-mueller-matrix "Link to this equation")\\[\begin{equation} \mathbf{s}''' = \mathbf{R}(\theta_{\text{out}}) \cdot \mathbf{s}'' = \underbrace{\mathbf{R}(\theta_{\text{out}}) \cdot \hat{\mathbf{M}} \cdot \mathbf{R}(\theta_{\text{in}})}_{\mathbf{M}} \cdot \mathbf{s} \end{equation}\\]
where \\(\theta_{\text{out}}\\) is the relative angle between the two frame bases \\(\hat{\mathbf{x}}_{\text{out}}\\) and \\(\mathbf{x}_{\text{out}}\\).
Effectively, this process transforms the matrix \\(\hat{\mathbf{M}}\\) into a modified matrix \\(\mathbf{M}\\) that operates between rotated incident and outgoing frames.
[![../../_images/rotated_mueller_matrix_crop.png](https://mitsuba.readthedocs.io/en/stable/_images/rotated_mueller_matrix_crop.png) ](https://mitsuba.readthedocs.io/en/stable/_images/rotated_mueller_matrix_crop.png)
**Figure 5** : Rotation of Mueller matrix incident and outgoing reference frames.[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rotated-mueller-matrix "Link to this image")
#### Rotation of optical elements[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rotation-of-optical-elements "Link to this heading")
Another common use case of the reference frame rotations is to find expressions for rotated optical elements. Consider [Figure 6](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rotated-element) for the following explanation. The optical element with Mueller matrix \\(\mathbf{M}\\) (valid for incident & outgoing frame (\\(\mathbf{x}', \mathbf{y}'\\))) is rotated by an angle \\(\theta\\). We now want to find the Mueller matrix \\(\mathbf{M}(\theta)\\) of this rotated element, expressed for incident and outgoing frame (\\(\mathbf{x}, \mathbf{y}\\)) that is aligned with the optical table. We again achieve this using two rotations, and the steps are analogous to the description above (Section [Rotation of Mueller matrix frames](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rotation-of-mueller-matrix-frames)):
First we rotate the incident Stokes vector to be expressed in the rotated frame:
(7)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq7 "Link to this equation")\\[\begin{equation} \mathbf{s}' = \mathbf{R}(\theta) \cdot \mathbf{s} \end{equation}\\]
We then apply the matrix \\(\mathbf{M}\\), as it is valid for the frame (\\(\mathbf{x}', \mathbf{y}'\\)):
(8)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq8 "Link to this equation")\\[\begin{equation} \mathbf{s}'' = \mathbf{M} \cdot \mathbf{s}' = \mathbf{M} \cdot \mathbf{R}(\theta) \cdot \mathbf{s} \end{equation}\\]
Finally, we rotate the resulting Stokes vector back into the original frame (\\(\mathbf{x}, \mathbf{y}\\)):
(9)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-rotated-element "Link to this equation")\\[\begin{equation} \mathbf{s}''' = \mathbf{R}(-\theta) \cdot \mathbf{s}'' = \underbrace{\mathbf{R}(-\theta) \cdot \mathbf{M} \cdot \mathbf{R}(\theta)}_{\mathbf{M}(\theta)} \cdot \mathbf{s} \end{equation}\\]
[![../../_images/rotated_element_crop.png](https://mitsuba.readthedocs.io/en/stable/_images/rotated_element_crop.png) ](https://mitsuba.readthedocs.io/en/stable/_images/rotated_element_crop.png)
**Figure 6** : An optical element rotated around the axis of propagation.[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rotated-element "Link to this image")
  
**Example 1: A rotated linear polarizer**
A very common special case of this is a linear polarizer rotated at some angle. Recall the Mueller matrix of a linear polarizer:
(10)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-linear-polarizer "Link to this equation")\\[\begin{split}\begin{equation} \mathbf{L} = \frac{1}{2} \begin{bmatrix} 1 & 1 & 0 & 0 \\\ 1 & 1 & 0 & 0 \\\ 0 & 0 & 0 & 0 \\\ 0 & 0 & 0 & 0 \end{bmatrix} \end{equation}\end{split}\\]
When applying Eq. [(9)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-rotated-element) to Eq. [(10)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-linear-polarizer) we get the expression
(11)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-linear-polarizer-rotated "Link to this equation")\\[\begin{split}\begin{equation} \mathbf{L}(\theta) = \frac{1}{2} \begin{bmatrix} 1 & \cos(2\theta) & \sin(2\theta) & 0 \\\ \cos(2\theta) & \cos^2(2\theta) & \sin(2\theta)\cos(2\theta) & 0 \\\ \sin(2\theta) & \sin(2\theta)\cos(2\theta) & \sin^2(2\theta) & 0 \\\ 0 & 0 & 0 & 0 \end{bmatrix} \end{equation}.\end{split}\\]
  
**Example 2: Malus’ law**
Consider a beam of unpolarized light (purple) that interacts with two linear polarizers as illustrated here:
The first polarizer transforms the beam into fully horizontally polarized light (middle) before traveling through a second polarizer at an angle \\(\theta\\). It is clear that a horizontal orientation (\\(\theta=0˚\\)) will allow all of the light to transmit as both polarizers are aligned with each other. Similarly, at an orthogonal configuration (\\(\theta=90˚\\)), all light will be absorbed. For intermediate angles, only a fraction of the light is transmitted in form of linearly polarized light. Under such a setting, the total intensities (first component of the Stokes vector) before and after interacting with the two polarizers, \\(\mathbf{s}_0, \mathbf{s}_0'\\), follow _Malus’ law_ [[[5]](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id21)]:
(12)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-malus-law "Link to this equation")\\[\begin{equation} \mathbf{s}_0' = \frac{\cos^2(\theta)}{2} \cdot \mathbf{s}_0 \end{equation}\\]
where \\(\theta\\) is the rotation angle of the second polarizer, measured counter-clockwise from the horizontal configuration.
This can easily be derived by in the Mueller-Stokes calculus, simply by left multiplying the matrices in Eq. [(10)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-linear-polarizer) and Eq. [(11)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-linear-polarizer-rotated) onto some arbitrary Stokes vector \\(\mathbf{s}\\):
\\[\begin{split}\begin{align*} \begin{bmatrix} \mathbf{s}_0' \\\ \mathbf{s}_1' \\\ \mathbf{s}_2' \\\ \mathbf{s}_3' \end{bmatrix} &= \mathbf{L}(\theta) \cdot \mathbf{L} \cdot \begin{bmatrix} \mathbf{s}_0 \\\ \mathbf{s}_1 \\\ \mathbf{s}_2 \\\ \mathbf{s}_3 \end{bmatrix} \\\ &= \frac{1}{4} \begin{bmatrix} 1 & \cos(2\theta) & \sin(2\theta) & 0 \\\ \cos(2\theta) & \cos^{2}(2\theta) & \sin(2\theta)\cos(2\theta) & 0 \\\ \sin(2\theta) & \sin(2\theta)\cos(2\theta) & \sin^{2}(2\theta) & 0 \\\ 0 & 0 & 0 & 0 \end{bmatrix} \cdot \begin{bmatrix} 1 & 1 & 0 & 0 \\\ 1 & 1 & 0 & 0 \\\ 0 & 0 & 0 & 0 \\\ 0 & 0 & 0 & 0 \end{bmatrix} \cdot \begin{bmatrix} \mathbf{s}_0 \\\ \mathbf{s}_1 \\\ \mathbf{s}_2 \\\ \mathbf{s}_3 \end{bmatrix} \\\ &= \frac{1}{4} \begin{bmatrix} 1 & \cos(2\theta) & \sin(2\theta) & 0 \\\ \cos(2\theta) & \cos^{2}(2\theta) & \sin(2\theta)\cos(2\theta) & 0 \\\ \sin(2\theta) & \sin(2\theta)\cos(2\theta) & \sin^{2}(2\theta) & 0 \\\ 0 & 0 & 0 & 0 \end{bmatrix} \cdot \begin{bmatrix} \mathbf{s}_0 + \mathbf{s}_1 \\\ \mathbf{s}_0 + \mathbf{s}_1 \\\ 0 \\\ 0 \end{bmatrix} \\\ &= \frac{1}{4} \begin{bmatrix} (\mathbf{s}_0 + \mathbf{s}_1) (1 + \cos(2\theta)) \\\ (\mathbf{s}_0 + \mathbf{s}_1) (1 + \cos(2\theta))\cos(2\theta) \\\ (\mathbf{s}_0 + \mathbf{s}_1) (1 + \cos(2\theta))\sin(2\theta) \\\ 0 \end{bmatrix} \\\ &= \frac{1}{2} \begin{bmatrix} (\mathbf{s}_0 + \mathbf{s}_1) (\cos^{2}\theta) \\\ (\mathbf{s}_0 + \mathbf{s}_1) (\cos^{2}\theta)\cos(2\theta) \\\ (\mathbf{s}_0 + \mathbf{s}_1) (\cos^{2}\theta)\sin(2\theta) \\\ 0 \end{bmatrix} \end{align*}\end{split}\\]
Here, the last step used the trigonometric identity \\(\cos(2\theta) = 2\cos^{2}\theta - 1\\). Plugging in the Stokes vector of unpolarized light (\\(\mathbf{s}_0 = 1, \mathbf{s}_1=\mathbf{s}_2=\mathbf{s}_3=0\\)) directly gives Eq. [(12)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-malus-law) as result.
  
**Example 3: Creation of circular polarization**
We can also use Mueller-Stokes calculus to understand a common physical setup to create circularly polarized light that uses a clever combination of a linear polarizer and a quarter-wave plate:
Any type of light (e.g. unpolarized, visualized in purple) is linearly polarized at a 45˚ angle by the first filter (left) and then hits a quarter-wave plate (green) that has its “fast axis” in a horizontal configuration. The wave plate introduces a quarter wavelength phase shift that slows down the vertical component (red). As a result, the two wave components are no longer aligned and the light beam is (perfectly) left-circularly polarized.
Or written in Mueller-Stokes calculus, using the quarter-wave plate from [Table 2](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#table-optical-elements) and the rotated linear polarizer from Eq. [(11)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-linear-polarizer-rotated) using \\(\theta=45˚\\):
\\[\begin{split}\begin{align*} \begin{bmatrix} \mathbf{s}_0' \\\ \mathbf{s}_1' \\\ \mathbf{s}_2' \\\ \mathbf{s}_3' \end{bmatrix} &= \frac{1}{2} \begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \\\ 0 & 0 & 0 & 1 \\\ 0 & 0 & -1 & 0 \end{bmatrix} \cdot \begin{bmatrix} 1 & \cos(2\theta) & \sin(2\theta) & 0 \\\ \cos(2\theta) & \cos^{2}(2\theta) & \sin(2\theta)\cos(2\theta) & 0 \\\ \sin(2\theta) & \sin(2\theta)\cos(2\theta) & \sin^{2}(2\theta) & 0 \\\ 0 & 0 & 0 & 0 \end{bmatrix} \cdot \begin{bmatrix} \mathbf{s}_0 \\\ \mathbf{s}_1 \\\ \mathbf{s}_2 \\\ \mathbf{s}_3 \end{bmatrix} \\\ &= \frac{1}{2} \begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \\\ 0 & 0 & 0 & 1 \\\ 0 & 0 & -1 & 0 \end{bmatrix} \cdot \begin{bmatrix} \mathbf{s}_0 + \mathbf{s}_2 \\\ 0 \\\ \mathbf{s}_0 + \mathbf{s}_2 \\\ 0 \end{bmatrix} \\\ &= \frac{1}{2} \begin{bmatrix} \mathbf{s}_0 + \mathbf{s}_2 \\\ 0 \\\ 0 \\\ -(\mathbf{s}_0 + \mathbf{s}_2) \end{bmatrix} \end{align*}\end{split}\\]
To figure out the signs of the final Stokes vector, recall that \\(\mathbf{s}_0 \geq \sqrt{\mathbf{s}_1^2 + \mathbf{s}_2^2 + \mathbf{s}_3^2}\\). From this we have \\((\mathbf{s}_0 + \mathbf{s}_2) > 0\\) and thus the last entry has to be negative which matches the observed left-circular polarization above.
* * *
## Fresnel equations[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#fresnel-equations "Link to this heading")
Specular dielectrics and conductors are important building blocks of material models in most rendering systems today. They are usually based on the well known _Fresnel equations_ (first derived by Augustin Jean Fresnel in the 19th century [[Fre23](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id32 "Augustin Jean Fresnel. Mémoire sur la loi des modifications que la réflexion imprime à la lumière polarisée. Académie des Sciences, 1823.")]) that describe the complete polarization state of specular reflection and refraction on such materials. Note that this is one of the few cases where exact expressions are available and therefore we definitely want to accurately capture this effect in polarization aware rendering.
At the core of this are of course the laws of reflection and refraction (a.k.a. Snell’s law)
(13)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-reflection-refraction-laws "Link to this equation")\\[\begin{split}\begin{align} \textbf{Reflection}: \;\;& \theta_i = \theta_r \\\ \textbf{Refraction}: \;\;& \eta_i \cdot \sin\theta_i = \eta_t \cdot \sin\theta_t \end{align}\end{split}\\]
that relate an incident angle \\(\theta_i\\) with its reflected (\\(\theta_r\\)) and refracted (\\(\theta_t\\)) analogues based on the indices of refraction (\\(\eta_i \text{ and } \eta_t\\)) on the two sides of the interface:
[![../../_images/reflection_transmission.svg](https://mitsuba.readthedocs.io/en/stable/_images/reflection_transmission.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/reflection_transmission.svg)
A typical example would be light that refracts from air into a denser medium such as water. In this case, we have \\(\eta_i \approx 1.0\\) and \\(\eta_t \approx 1.33\\).
### Theory[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#theory "Link to this heading")
Modern formulations of the Fresnel equations are based on electromagnetic theory [[[6]](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id42)] where we consider the two transverse fields \\(E\\) (electric) and \\(H\\) (magnetic). Usually, derivations relate the incident (\\(E_i\\)) with the reflected (\\(E_r\\)) and transmitted (\\(E_t\\)) electric fields, though equivalently the magnetic fields could also be used. It is sufficient to consider two cases where the electric field arrives either in perpendicular (”\\(\bot\\)”) or parallel (”\\(\parallel\\)”) orientation relative to the plane of incidence [[[7]](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id43)]:
To describe these fields, we have to make some choices regarding coordinate systems. In particular: in which directions should the electric fields be pointing before and after interacting with the interface? Given the electric field direction \\(E\\) and the propagation direction of the beam \\(\mathbf{z}\\), the orientation of the magnetic field \\(H\\) is always clearly defined by a right-handed coordinate system \\(E \times H = \alpha \cdot \mathbf{z}\\) for some constant \\(\alpha\\) (Poynting’s theorem). The orientation of the electric field itself is however somewhat arbitrary. Incident and reflected field could for instance point in the same or opposite directions of each other.
Warning
**Electric field conventions**
Unfortunately, different choices for the orientations of the electric field will result in slightly different versions of the final Fresnel equations. All versions are “correct” however as long as conventions are clear and consistent. We therefore clarify the concrete alignments that we use in the following diagram:
[![../../_images/electric_magnetic_fields_verdet.svg](https://mitsuba.readthedocs.io/en/stable/_images/electric_magnetic_fields_verdet.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/electric_magnetic_fields_verdet.svg)
**Figure 7** : The convention we use for directions of the electric (\\(E\\)) and magnetic (\\(H\\)) fields in case of specular reflection and transmission. In this configuration, all electric fields are parallel to each other which also known as the _Verdet convention_ in the literature.[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#electric-magnetic-fields-verdet-first "Link to this image")
**Perpendicular electric field**
In this case ([Figure 7](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#electric-magnetic-fields-verdet-first) (**a**)) all electric field vectors \\(E_i^{\bot}, E_r^{\bot}\\), and \\(E_t^{\bot}\\) are collinear. It is therefore the most natural to orient them all in the same direction, e.g. pointing into the screen / plane. In fact, practically all sources agree on this convention [[Muller69](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id33 "Rolf H. Muller. Definitions and conventions in ellipsometry. Surface Science, 16:14-33, August 1969. doi:10.1016/0039-6028\(69\)90003-X.")].
**Parallel electric field**
Here, the three vectors \\(E_i^{\bot}, E_r^{\bot}\\), and \\(E_t^{\bot}\\) are coplanar but (for general \\(\theta_i\\)) not collinear anymore and it is therefore not so clear what it means to “point into the same direction”. Instead, this is now defined based on the magnetic field vectors \\(H_i^{\bot}, H_r^{\bot}\\), and \\(H_t^{\bot}\\) which are collinear.
In [Figure 7](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#electric-magnetic-fields-verdet-first) (**b**), we decided to have them all point into the same direction again which is commonly referred to as the _Verdet convention_ in the literature. Most sources agree on the directions of \\(E_i^{\parallel}\\) vs. \\(E_t^{\parallel}\\) in the transmission case. However a large number of them choose to orient \\(E_r^{\parallel}\\) (and \\(H_r^{\parallel}\\)) in the opposite direction [[Muller69](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id33 "Rolf H. Muller. Definitions and conventions in ellipsometry. Surface Science, 16:14-33, August 1969. doi:10.1016/0039-6028\(69\)90003-X.")], also known as the _Fresnel convention_. Such a sign flip will propagate all the way into the signs of the Fresnel equations but it is important to keep in mind that both are “correct” based on the chosen coordinate systems. See Section [Different conventions in the literature](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#different-conventions-in-the-literature) below for an extended discussion.
#### Equations[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equations "Link to this heading")
Based on the chosen convention above, we now summarize the Fresnel equations as implemented in Mitsuba 3. For the purpose of this document we omit their actual derivations and instead refer interested readers to _Optics_ by E. Hecht [[Hec98](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id27 "Eugene Hecht. Optics. Addison-Wesley, 4th edition, 1998.")].
The _reflection amplitude coefficients_ for the “\\(\bot\\)” and “\\(\parallel\\)” components relate the incident and reflected electric fields via
(14)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-reflection-amplitudes-s "Link to this equation")\\[\begin{align} r_{\bot} = \frac{E_r^{\bot}}{E_i^{\bot}} = \frac{\eta_i \cos\theta_i - \eta_t \cos\theta_t}{\eta_i \cos\theta_i + \eta_t \cos\theta_t} = -\frac{\sin(\theta_i - \theta_t)}{\sin(\theta_i + \theta_t)} \end{align}\\]
(15)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-reflection-amplitudes-p "Link to this equation")\\[\begin{align} r_{\parallel} = \frac{E_r^{\parallel}}{E_i^{\parallel}} = \frac{\eta_t \cos\theta_i - \eta_i \cos\theta_t}{\eta_t \cos\theta_i + \eta_i \cos\theta_t} = +\frac{\tan(\theta_i - \theta_t)}{\tan(\theta_i + \theta_t)} \end{align}\\]
where \\(\cos\theta_t\\) can be computed from Snell’s law (Eq. [(13)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-reflection-refraction-laws)) as \\(\cos\theta_t = \sqrt{1 - {\left(\frac{\eta_i}{\eta_t}\right)}^{2} {\sin}^{2}\theta_i}\\). A negative value under the square root in this expression usually indicates that refraction is impossible, and instead all light is reflected (the _total internal reflection_ case) but here, the complex valued square root is used as part of Eq. [(14)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-reflection-amplitudes-s) and Eq. [(15)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-reflection-amplitudes-p).
Note that the leading signs of both equations would change based on different conventions for the electric field orientations above.
The same expressions also hold in the conductor case where the indices of refraction are complex valued, i.e. \\(\eta = n - k i\\) with real and imaginary parts \\(n\\) and \\(k\\). The use of complex values here is a mathematical trick to encode both the harmonic oscillation of a wave together with its decay when travelling into the conductive medium. As illustration, consider a wave travelling along spatial coordinate \\(x\\) and time \\(t\\):
\\[\exp\left(i \omega (t - \frac{\eta}{c} x)\right) = \exp\left(i \omega (t - \frac{n}{c} x)\right) \cdot \exp\left(-\omega \frac{k}{c} x\right)\\]
In this context, \\(\omega\\) denotes the angular spatial frequency and \\(c\\) is the speed of light. A value \\(k > 0\\) introduces an exponential falloff term which is why \\(k\\) is also known as the _extinction coefficient_. We refer to _“Optical Properties of Metals”_ in Hecht [[Hec98](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id27 "Eugene Hecht. Optics. Addison-Wesley, 4th edition, 1998.")] (Section 4.8 of the 5th edition) for a derivation and further details.
There also exist alternate conventions of the above behaviour involving a complex conjugate, i.e. describing the wave as \\(\exp\left(i\omega(\frac{\eta}{c}x - t)\right)\\) instead. As a consequence, the sign of the imaginary part needs to flip to \\(n + k i\\) accordingly. These are sometimes referred to as the “\\(\exp(+i \omega t)\\) and \\(\exp(-i \omega t)\\) conventions” in the literature [[Muller69](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id33 "Rolf H. Muller. Definitions and conventions in ellipsometry. Surface Science, 16:14-33, August 1969. doi:10.1016/0039-6028\(69\)90003-X.")]. In this document we use the former option, so \\(\eta = n - k i\\).
* * *
Similarly, we can use the complex amplitudes after refraction to determine the _transmission amplitude coefficients_ , where a few different expressions are commonly found:
(16)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq16 "Link to this equation")\\[\begin{align} t_{\bot} &= \frac{E_t^{\bot}}{E_i^{\bot}} = \frac{2 \cos\theta_i \sin\theta_t}{\sin(\theta_i + \theta_t)} = \frac{2 \eta_i \cos\theta_i}{\eta_i \cos\theta_i + \eta_t \cos\theta_t} \end{align}\\]
(17)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq17 "Link to this equation")\\[\begin{align} t_{\parallel} &= \frac{E_t^{\parallel}}{E_i^{\parallel}} = \frac{2 \cos\theta_i \sin\theta_t}{\sin(\theta_i + \theta_t) \cos(\theta_i - \theta_t)} = \frac{2 \eta_i \cos\theta_i}{\eta_t \cos\theta_i + \eta_i \cos\theta_t} \end{align}\\]
where, as mentioned above, most sources seem to agree on the signs. A particular simple expression of these is also available in case the reflection coefficients have already been computed:
(18)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq18 "Link to this equation")\\[\begin{align} t_{\bot} = 1 + r_{\bot}, \;\;\;\;\; t_{\parallel} = (1 + r_{\parallel}) \frac{\eta_i}{\eta_t} \end{align}\\]
Obviously, these last expression now depend on the choice of the electric field directions again.
* * *
The complex reflection amplitudes also encode the _phase shifts_ of the respective wave components. The relative phase shift between “\\(\parallel\\)” and “\\(\bot\\)” components is given by
(19)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-phase-shifts "Link to this equation")\\[\begin{align} \Delta = \delta_\parallel - \delta_\bot = \arg(r_\parallel) - \arg(r_\bot) \end{align}\\]
where both \\(\delta_\parallel\\) and \\(\delta_\bot\\) (when positive valued) are phase _advances_ when time is measured to increase in the positive direction [[Muller69](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id33 "Rolf H. Muller. Definitions and conventions in ellipsometry. Surface Science, 16:14-33, August 1969. doi:10.1016/0039-6028\(69\)90003-X.")].
The transmission amplitudes \\(t_\bot\\) and \\(t_\parallel\\) are always real valued and refraction therefore does not cause any phase shifts.
* * *
Two more important quantities are the _reflectance_ (R) and _transmittance_ (T):
(20)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq20 "Link to this equation")\\[\begin{align} R = \frac{\text{reflected power}}{\text{incident power}} = \frac{A_r I_r}{A_i I_i} \end{align}\\]
(21)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq21 "Link to this equation")\\[\begin{align} T = \frac{\text{transmitted power}}{\text{incident power}} = \frac{A_t I_t}{A_i I_i} \end{align}\\]
where \\(A\\) is the beam’s area and \\(I\\) is the intensity
(22)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq22 "Link to this equation")\\[\begin{align} I = \eta \frac{c_0 \, \epsilon_0}{2} {|E|}^{2} \end{align}\\]
with refractive index \\(\eta\\), speed of light (in vacuum) \\(c_0\\), vacuum permittivity \\(\epsilon_0\\), and electric field \\(E\\).
The beam area ratios are found by simple trigonometry to be \\(\frac{A_r}{A_i} = \frac{\cos\theta_i}{\cos\theta_i} = 1\\) and \\(\frac{A_t}{A_i} = \frac{\cos\theta_t}{\cos\theta_i}\\):
[![../../_images/reflectance_transmittance.svg](https://mitsuba.readthedocs.io/en/stable/_images/reflectance_transmittance.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/reflectance_transmittance.svg)
from which we have the final expressions
(23)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-reflectance "Link to this equation")\\[\begin{align} R = \frac{\eta_i {|E_r|}^{2}}{\eta_i {|E_i|}^{2}} = |r|^{2} \end{align}\\]
(24)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-transmittance "Link to this equation")\\[\begin{align} T = \frac{\cos\theta_t \eta_t {|E_t|}^{2}}{\cos\theta_i \eta_i {|E_i|}^{2}} = \frac{\cos\theta_t}{\cos\theta_i } \frac{\eta_t}{\eta_i} |t|^{2} \end{align}\\]
From energy conservation, we always have \\(R_{\bot} + T_{\bot} = 1\\) and \\(R_{\parallel} + T_{\parallel}\\) = 1, even though the same does not always hold for the amplitudes \\(r\\) and \\(t\\).
#### Different conventions in the literature[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#different-conventions-in-the-literature "Link to this heading")
When considering all possible orientations of the electric field vectors (2 choices for \\(E_i\\), \\(E_r\\), and \\(E_t\\) for both “\\(\bot\\)” and “\\(\parallel\\)”) there seem to 16 different versions. Luckily, many of them are equivalent and from the few remaining ones, the literature is mostly divided into two groups which orient the reflected electric field vector \\(E_r^{\parallel}\\) differently:
1. **Fresnel convention**
  

Here, \\(E_i^{\parallel}\\) and \\(E_r^{\parallel}\\) are oriented s.t. their magnetic fields \\(H_i^{\bot}\\) and \\(H_r^{\bot}\\) point in _opposite_ directions ([Figure 8](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#electric-magnetic-fields-fresnel) below). Its name comes from the fact that it is close to Fresnel’s original description of the equations where both perpendicular and parallel amplitudes (Eq. [(14)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-reflection-amplitudes-s) and Eq. [(15)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-reflection-amplitudes-p)) would use the same sign.
  

Its main appeal is that at perpendicular incidence, both \\(E_i^{\parallel}\\) and \\(E_r^{\parallel}\\) are indistinguishable which makes sense from a physical perspective. However, it accomplishes this by having \\(E_r^{\bot}\\) & \\(E_r^{\parallel}\\) form a left handed coordinate system which can be less convenient for subsequent calculations.
[![../../_images/electric_magnetic_fields_fresnel.svg](https://mitsuba.readthedocs.io/en/stable/_images/electric_magnetic_fields_fresnel.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/electric_magnetic_fields_fresnel.svg)
**Figure 8** : The _Fresnel convention_ is a common alternative of orienting the electric fields. The difference to the _Verdet convention_ is a change of handedness of the \\(H_r^{\bot}\\) and \\(E_r^{\parallel}\\) frame of the reflected direction highlighted in red.[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#electric-magnetic-fields-fresnel "Link to this image")
2. **Verdet convention**
  

Here, \\(E_i^{\parallel}\\) and \\(E_r^{\parallel}\\) are oriented s.t. their magnetic fields \\(H_i^{\bot}\\) and \\(H_r^{\bot}\\) point in _the same_ direction ([Figure 7](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#electric-magnetic-fields-verdet-first) and repeated below as [Figure 9](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#electric-magnetic-fields-verdet) for easier comparison). It is named after Verdet, who was an editor of Fresnel and originally switched the convention because in his opinion, the two fields should instead be equivalent at grazing angles [[PP97](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id37 "Edward D Palik and Edward J Prucha. Handbook of optical constants of solids. Academic Press, Boston, MA, 1997. URL: https://cds.cern.ch/record/396087.")], [[Cla09](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id28 "David Clarke. Stellar Polarimetry, Appendix A: The Fresnel Laws. John Wiley & Sons, Ltd, 2009. ISBN 9783527628322. doi:https://doi.org/10.1002/9783527628322.app1.")]. Compared to Fresnel’s original formulation of the equations, the sign of the parallel amplitude consequently had to be flipped, see Eq. [(15)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-reflection-amplitudes-p).
[![../../_images/electric_magnetic_fields_verdet.svg](https://mitsuba.readthedocs.io/en/stable/_images/electric_magnetic_fields_verdet.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/electric_magnetic_fields_verdet.svg)
**Figure 9** : The convention we use for directions of the electric (\\(E\\)) and magnetic (\\(H\\)) fields in case of specular reflection and transmission. In this configuration, all electric fields are parallel to each other which also known as the _Verdet convention_ in the literature.[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#electric-magnetic-fields-verdet "Link to this image")
Because the conventions involve changing the coordinate system orientations, the sign change between the resulting variants of the Fresnel equations can also just be interpreted as phase shift of \\(180˚\\). However we should emphasize again that, despite their different formulations, all conventions are ultimately correct in their frame of reference — as long as they are clearly defined and used consistently.
* * *
Apart from these different coordinate systems, and the few other conventions we mentioned before (e.g. time dependence of the waves vs. the complex index of refraction in Section [Equations](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equations)) there are also numerous other aspects of polarization that are not universally agreed upon. We found the the 1969 paper by Muller [[Muller69](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id33 "Rolf H. Muller. Definitions and conventions in ellipsometry. Surface Science, 16:14-33, August 1969. doi:10.1016/0039-6028\(69\)90003-X.")] very useful to understand the space of possibilities and we adopt all conventions recommended in that article. (More precisely the final variants suggested by H. E. Bennett in the discussion section at the end.) Other useful resource to us were [[Hec98](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id27 "Eugene Hecht. Optics. Addison-Wesley, 4th edition, 1998.")], [[PP97](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id37 "Edward D Palik and Edward J Prucha. Handbook of optical constants of solids. Academic Press, Boston, MA, 1997. URL: https://cds.cern.ch/record/396087.")], [[BDE+09](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id38 "Michael Bass, Casimer DeCusatis, Jay Enoch, Vasudevan Lakshminarayanan, Guifang Li, Carolyn Macdonald, Virendra Mahajan, and Eric Van Stryland. Handbook of Optics, Third Edition Volume I: Geometrical and Physical Optics, Polarized Light, Components and Instruments\(Set\). McGraw-Hill, Inc., USA, 3 edition, 2009. ISBN 0071498893.")], [[HS80](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id39 "R. H. Muller Hauge and C. G. Smith. Conventions and formulas for using the mueller-stokes calculus in ellipsometry. Surface Science, 1980.")], [[FS65](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id40 "G. Friedmann and H. S. Sandhu. Phase change on reflection from isotropic dielectrics. American Journal of Physics, 33\(2\):135-138, 1965. doi:10.1119/1.1971270.")], [[OV20](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id41 "Minsu Oh and Thomas Vandervelde. Bridging the gaps between different sign conventions of fresnel reflection coefficients towards a universal form. In 2020 IEEE 63rd International Midwest Symposium on Circuits and Systems \(MWSCAS\), volume, 707-713. 2020. doi:10.1109/MWSCAS48704.2020.9184440.")], [[Salik12](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id42 "Ertan Salik. Quantitative investigation of Fresnel reflection coefficients by polarimetry. American Journal of Physics, 80\(3\):216-224, March 2012. doi:10.1119/1.3672851.")], [[Azz04](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id35 "R. M. A. Azzam. Phase shifts that accompany total internal reflection at a dielectric–dielectric interface. J. Opt. Soc. Am. A, 21\(8\):1559–1563, Aug 2004. URL: https://josaa.osa.org/abstract.cfm?URI=josaa-21-8-1559, doi:10.1364/JOSAA.21.001559.")].
Ultimately, each convention has its own justifications and use cases in different branches of science and a lack of consistency is thus understandable. Nonetheless it is an unfortunate circumstance, especially for anyone new to the theory of polarization.
* * *
### Analysis and Mueller matrix formulation[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#analysis-and-mueller-matrix-formulation "Link to this heading")
We now turn to discussing the Fresnel equations in the various cases of reflection and transmission for dielectrics and conductors. At the same time, we will state the corresponding Mueller matrices that encode the laws as part of the Mueller-Stokes calculus used in the renderer. This conversion (when following all conventions from above) is straightforward and was first discussed by Collett [[Col71](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id29 "Edward Collett. Mueller-stokes matrix formulation of fresnel's equations. American Journal of Physics, 39\(5\):517-528, 1971. doi:10.1119/1.1986205.")]. At the end, in Section [Different conventions in the context of Mueller matrices](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#different-conventions-in-the-context-of-mueller-matrices) we will briefly return to some of the alternative conventions and how they sometimes affect Mueller matrix definitions in the literature.
#### Dielectric reflection (at a denser medium)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#dielectric-reflection-at-a-denser-medium "Link to this heading")
The first case is simple dielectric reflection at an interface that is denser than the incident medium. Consider for instance [Figure 10](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#reflection-from-outside) with \\(\eta_i = 1.0\\) and \\(\eta_t = 1.5\\). The curves with “\\(\bot\\)” and “\\(\parallel\\)” components of the reflectance is also well known in graphics where usually rendering systems implement the non-polarized version \\(R_{avg} = R_\bot + R_\parallel\\) that just averages the two curves.
[![../../_images/reflection_from_outside.svg](https://mitsuba.readthedocs.io/en/stable/_images/reflection_from_outside.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/reflection_from_outside.svg)
**Figure 10** : Reflectance, phase shifts, and degree of polarization (DOP) for varying incident angle of a specular reflection at the outside of a dielectric interface with relative IOR of \\(3/2\\) (\\(\eta_i = 1.0, \eta_t = 1.5\\)).[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#reflection-from-outside "Link to this image")
The angle \\(\theta = \arctan(\eta_t / \eta_i)\\), also known as _Brewster’s angle_ is of special importance here, as it has several interesting properties:
  1. The “\\(\parallel\\)” reflectance is zero.
  2. A phase shift of \\(180˚\\) takes place which is comparable to a half-wave plate. Note that such a discontinuous “jump” might at first seem implausible but the underlying physics is still continuous because of \\(R_\parallel\\) smoothly approaching zero at the same time.
  3. The light is fully polarized and oscillates along the “\\(\bot\\)” component (horizontally with respect to the reference frame (\\(\mathbf{x}, \mathbf{y}\\))).


The Mueller matrix for this case follows the shape of a standard polarizer (see [Table 2](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#table-optical-elements)) where we know how much energy is preserved at the two orthogonal components “\\(\bot\\)” (horizontal) and “\\(\parallel\\)” (vertical):
(25)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-mm-reflection-from-outside "Link to this equation")\\[\begin{split}\begin{equation} \frac{1}{2} \cdot \begin{bmatrix} R_{\bot} + R_{\parallel} & R_{\bot} - R_{\parallel} & 0 & 0 \\\ R_{\bot} - R_{\parallel} & R_{\bot} + R_{\parallel} & 0 & 0 \\\ 0 & 0 & 2 \sqrt{R_{\bot} R_{\parallel}} & 0 \\\ 0 & 0 & 0 & 2 \sqrt{R_{\bot} R_{\parallel}} \end{bmatrix} \end{equation}\end{split}\\]
#### Dielectric transmission (into a denser medium)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#dielectric-transmission-into-a-denser-medium "Link to this heading")
An analogous case is refraction into a medium of higher density as shown in [Figure 11](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#transmission-from-outside) for \\(\eta_i = 1.0\\) and \\(\eta_t = 1.5\\).
[![../../_images/transmission_from_outside.svg](https://mitsuba.readthedocs.io/en/stable/_images/transmission_from_outside.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/transmission_from_outside.svg)
**Figure 11** : Transmittance and phase shifts for varying incident angles of a specular refraction from vacuum (\\(\eta_i = 1.0\\)) into a dielectric (\\(\eta_t = 1.5\\)).[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#transmission-from-outside "Link to this image")
As discusses previously, there is no phase shift for transmission (the transmission amplitude coefficients are always real). The matrix is also analogous to above:
(26)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-mm-transmission "Link to this equation")\\[\begin{split}\begin{equation} \frac{1}{2} \cdot \begin{bmatrix} T_{\bot} + T_{\parallel} & T_{\bot} - T_{\parallel} & 0 & 0 \\\ T_{\bot} - T_{\parallel} & T_{\bot} + T_{\parallel} & 0 & 0 \\\ 0 & 0 & 2 \sqrt{T_{\bot} T_{\parallel}} & 0 \\\ 0 & 0 & 0 & 2 \sqrt{T_{\bot} T_{\parallel}} \end{bmatrix} \end{equation}\end{split}\\]
#### Dielectric reflection and transmission (from a denser to a less dense medium)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#dielectric-reflection-and-transmission-from-a-denser-to-a-less-dense-medium "Link to this heading")
Internal reflection _inside_ a dense dielectric (\\(\eta_t / \eta_i < 1\\)) is an interesting case due to the well known _critical angle_ (\\(\theta = \arcsin(\eta_t / \eta_i)\\)) after which all light is reflected and thus transmittance goes to zero. This is also known as _total internal reflection (TIR)_ See e.g. [Figure 12](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#reflection-from-inside) for the reflection case and [Figure 13](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#transmission-from-inside) for transmission in case of \\(\eta_i = 1.5\\) and \\(\eta_t = 1.0\\).
[![../../_images/reflection_from_inside.svg](https://mitsuba.readthedocs.io/en/stable/_images/reflection_from_inside.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/reflection_from_inside.svg)
**Figure 12** : Reflectance and phase shifts for varying incident angles of a specular reflection from the inside of a dielectric interface with relative IOR \\(2/3\\) (\\(\eta_i = 1.5, \eta_t = 1.0\\)). Compare also with Fig. 1 in [[Azz04](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id35 "R. M. A. Azzam. Phase shifts that accompany total internal reflection at a dielectric–dielectric interface. J. Opt. Soc. Am. A, 21\(8\):1559–1563, Aug 2004. URL: https://josaa.osa.org/abstract.cfm?URI=josaa-21-8-1559, doi:10.1364/JOSAA.21.001559.")].[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#reflection-from-inside "Link to this image")
[![../../_images/transmission_from_inside.svg](https://mitsuba.readthedocs.io/en/stable/_images/transmission_from_inside.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/transmission_from_inside.svg)
**Figure 13** : Transmittance and phase shifts for varying incident angles of a specular refraction from a dielectric (\\(\eta_i = 1.0\\)) into vacuum (\\(\eta_t = 1.5\\)).[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#transmission-from-inside "Link to this image")
The phase shifts in this case were studied in detail by Azzam [[Azz04](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id35 "R. M. A. Azzam. Phase shifts that accompany total internal reflection at a dielectric–dielectric interface. J. Opt. Soc. Am. A, 21\(8\):1559–1563, Aug 2004. URL: https://josaa.osa.org/abstract.cfm?URI=josaa-21-8-1559, doi:10.1364/JOSAA.21.001559.")]. Similar to the previous case in Section [Dielectric reflection (at a denser medium)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#dielectric-reflection-at-a-denser-medium), there is a phase change of \\(180˚\\) after \\(\theta = \arctan(\eta_t / \eta_i)\\) but now there is an additional variation in the phase shift for incident directions below the critical angle. It’s maximal value is located at angle
(27)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-tir-phase-minimum "Link to this equation")\\[\begin{equation} \arg \max_\theta \Delta(\theta) = \arccos\sqrt{\frac{1 - (\eta_t / \eta_i)^2}{1 + (\eta_t / \eta_i)^2}} \end{equation}\\]
Because all light is reflected (\\(R_\bot = R_\parallel = 1\\)) and \\(\Delta \neq 0\\), the Mueller matrix for the total internal reflection case is just a retarder:
(28)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-mm-tir "Link to this equation")\\[\begin{split}\begin{equation} \begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \\\ 0 & 0 & \cos\Delta & -\sin\Delta \\\ 0 & 0 & \sin\Delta & \cos\Delta \end{bmatrix} \end{equation}\end{split}\\]
The sign is flipped compared to the retarder matrix in [Table 2](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#table-optical-elements) which is a consequence of using the relative phase shift definition from [[Muller69](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id33 "Rolf H. Muller. Definitions and conventions in ellipsometry. Surface Science, 16:14-33, August 1969. doi:10.1016/0039-6028\(69\)90003-X.")] (Eq. [(19)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-phase-shifts)). See Section [Different conventions in the context of Mueller matrices](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#different-conventions-in-the-context-of-mueller-matrices) for formulations under alternative conventions.
**Example: Fresnel rhomb**
Fresnel [[Fre23](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id32 "Augustin Jean Fresnel. Mémoire sur la loi des modifications que la réflexion imprime à la lumière polarisée. Académie des Sciences, 1823.")] realized that the phase shifts due to total internal reflection can be used to turn linear into circular polarization by constructing a prism where light is is reflected twice on the interior of a dielectric material. The angle have to be chosen carefully based on the refractive index s.t. the correct phase shift of \\(90˚\\) (i.e. equivalent to a quarter-wave plate) is achieved:
[![../../_images/fresnel_rhomb.svg](https://mitsuba.readthedocs.io/en/stable/_images/fresnel_rhomb.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/fresnel_rhomb.svg)
Note that, as seen in [Figure 12](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#reflection-from-inside), a single reflection is not sufficient for typical glass-like materials as the retardation is actually \\(< 90˚\\). Instead, the two reflections will each cause a shift of \\(45˚\\). More precisely, the “\\(\parallel\\)” component will be advanced by \\(90˚\\) relative to “\\(\bot\\)”.
A specific example would be \\(\eta_i \approx 1.49661\\) where the correct shift (\\(\Delta = 45˚\\)) is achieved with \\(\theta_i = \arg \min_\theta \Delta(\theta) \approx 51.782˚\\) (Eq. [(27)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-tir-phase-minimum)).
Multiplying the Mueller matrices of the two reflections gives
\\[\begin{split}\begin{align} \mathbf{M}_{rhomb} &= \begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \\\ 0 & 0 & \cos\Delta & -\sin\Delta \\\ 0 & 0 & \sin\Delta & \cos\Delta \end{bmatrix} \cdot \begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \\\ 0 & 0 & \cos\Delta & -\sin\Delta \\\ 0 & 0 & \sin\Delta & \cos\Delta \end{bmatrix} \\\ &= \begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \\\ 0 & 0 & \cos^{2}\Delta - \sin^{2}\Delta & -2\cos\Delta\sin\Delta \\\ 0 & 0 & 2\cos\Delta\sin\Delta & \cos^{2}\Delta - \sin^{2}\Delta \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \\\ 0 & 0 & \cos(2\Delta) & -\sin(2\Delta) \\\ 0 & 0 & \sin(2\Delta) & \cos(2\Delta) \end{bmatrix} \\\ &= \begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \\\ 0 & 0 & 0 & -1 \\\ 0 & 0 & 1 & 0 \end{bmatrix} \end{align}\end{split}\\]
which is equivalent to a Mueller matrix of a quarter-wave plate with its fast axis vertical, i.e. aligned with “\\(\parallel\\)”, see [Table 2](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#table-optical-elements).
Finally, when sending linearly (diagonal at \\(+45˚\\)) polarized light through the prism, the outgoing light will be right-circularly polarized:
\\[\begin{split}\begin{align} \mathbf{M}_{rhomb} \cdot \begin{bmatrix} 1 \\\ 0 \\\ 1 \\\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\\ 0 \\\ 0 \\\ 1 \end{bmatrix} \end{align}\end{split}\\]
#### Conductor reflection[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#conductor-reflection "Link to this heading")
As mentioned at the beginning of Section [Theory](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#theory), all equations also generalize the the case of reflection on conductors that have complex valued indices of refraction \\(\eta = n - k i\\) where \\(n\\) is the real part and \\(k\\) is the extinction coefficient. [Figure 14](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#reflection-conductor) shows a typical case with \\(\eta = 0.183 - 3.43i\\) (gold at 633nm).
[![../../_images/reflection_conductor.svg](https://mitsuba.readthedocs.io/en/stable/_images/reflection_conductor.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/reflection_conductor.svg)
**Figure 14** : Reflectance and phase shifts for varying incident angles of a specular reflection on a conductor with complex IOR of \\(\eta = 0.183 - 3.43i\\). Compare also with Fig. B in [[Muller69](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id33 "Rolf H. Muller. Definitions and conventions in ellipsometry. Surface Science, 16:14-33, August 1969. doi:10.1016/0039-6028\(69\)90003-X.")].[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#reflection-conductor "Link to this image")
Plugging in complex IORs with the wrong sign convention into the equations here will result in a sign flip between the two \\(\sin(...)\\) expressions in the matrix below. To avoid any issues in our implementation, Mitsuba 3 will automatically flip the sign appropriately so both conventions of input parameters can be used.
Compared to the dielectric cases earlier, every incident angle results in some phase shift now, and thus the Mueller matrix is a combination of a polarizer and retarder:
(29)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-mm-conductor "Link to this equation")\\[\begin{split}\begin{equation} \frac{1}{2} \begin{bmatrix} R_{\bot} + R_{\parallel} & R_{\bot} - R_{\parallel} & 0 & 0 \\\ R_{\bot} - R_{\parallel} & R_{\bot} + R_{\parallel} & 0 & 0 \\\ 0 & 0 & 2 \sqrt{R_{\bot} R_{\parallel}} \cos\Delta & -2 \sqrt{R_{\bot} R_{\parallel}} \sin\Delta \\\ 0 & 0 & 2 \sqrt{R_{\bot} R_{\parallel}} \sin\Delta & 2 \sqrt{R_{\bot} R_{\parallel}} \cos\Delta \end{bmatrix} \end{equation}\end{split}\\]
The dashed line in [Figure 14](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#reflection-conductor) also illustrates the _principle angle of incidence_ which is \\(\theta_i\\) where the relative phase shift is exactly a quarter wavelength (\\(90˚\\)). It can be used to measure the complex IOR of a metallic material [[Col05](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id30 "E. Collett. Field Guide to Polarization. Field Guide Series. Society of Photo Optical, 2005. ISBN 9780819458681.")].
#### Fully general case (dielectric + conductor) covering all cases[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#fully-general-case-dielectric-conductor-covering-all-cases "Link to this heading")
Finally, we can summarize all cases using just two different Mueller matrices for reflection and transmission respectively which makes an implementation less error-prone:
(30)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-mm-general-reflection "Link to this equation")\\[\begin{split}\begin{equation} \mathbf{F}_r = \frac{1}{2} \cdot \begin{bmatrix} R_{\bot} + R_{\parallel} & R_{\bot} - R_{\parallel} & 0 & 0 \\\ R_{\bot} - R_{\parallel} & R_{\bot} + R_{\parallel} & 0 & 0 \\\ 0 & 0 & 2 \sqrt{R_{\bot} R_{\parallel}} \cos\Delta & -2 \sqrt{R_{\bot} R_{\parallel}} \sin\Delta \\\ 0 & 0 & 2 \sqrt{R_{\bot} R_{\parallel}} \sin\Delta & 2 \sqrt{R_{\bot} R_{\parallel}} \cos\Delta \end{bmatrix} \end{equation}\end{split}\\]
(31)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq-mm-general-transmission "Link to this equation")\\[\begin{split}\begin{equation} \mathbf{F}_t = \frac{1}{2} \cdot \begin{bmatrix} T_{\bot} + T_{\parallel} & T_{\bot} - T_{\parallel} & 0 & 0 \\\ T_{\bot} - T_{\parallel} & T_{\bot} + T_{\parallel} & 0 & 0 \\\ 0 & 0 & 2 \sqrt{T_{\bot} T_{\parallel}} & 0 \\\ 0 & 0 & 0 & 2 \sqrt{T_{\bot} T_{\parallel}} \end{bmatrix} \end{equation}\end{split}\\]
It can be seen how Eq. [(30)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-mm-general-reflection) simplifies back to the simpler case earlier (Eq. [(25)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-mm-reflection-from-outside)) in case the index of refraction is real (dielectric) and no total internal reflection occurs. In that case, the relative phase shift \\(\Delta = 0\\), so \\(\cos\Delta = 1\\) and \\(\sin\Delta = 0\\).
#### Different conventions in the context of Mueller matrices[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#different-conventions-in-the-context-of-mueller-matrices "Link to this heading")
Some of the conventions used in other sources of course also affect the Mueller matrix formulations. As a result, the matrices written above exist in the literature in various versions that differ in the signs of the individual entries.
**1. Fresnel vs. Verdet convention**
In the Fresnel convention (Section [Different conventions in the literature](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#different-conventions-in-the-literature)), the reflected field vectors \\(E_r^{\bot}\\) and \\(E_r^{\parallel}\\) define a left handed coordinate system with the direction of propagation which is incompatible with the usual definitions of the Stokes parameters. As a workaround, in case this convention is used for the Fresnel equations, an additional handedness change matrix
(32)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq32 "Link to this equation")\\[\begin{split}\begin{equation} \begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \\\ 0 & 0 & -1 & 0 \\\ 0 & 0 & 0 & -1 \end{bmatrix} \end{equation}\end{split}\\]
needs to be introduced [[Cla09](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id28 "David Clarke. Stellar Polarimetry, Appendix A: The Fresnel Laws. John Wiley & Sons, Ltd, 2009. ISBN 9783527628322. doi:https://doi.org/10.1002/9783527628322.app1.")]. Effectively, this is nothing different than the Mueller matrix of a half-wave plate ([Table 2](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#table-optical-elements)) that introduces a \\(180˚\\) phase shift that switches all signs to the Verdet convention.
This would affect all matrices involving reflections (Eq. [(25)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-mm-reflection-from-outside), Eq. [(28)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-mm-tir), Eq. [(29)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-mm-conductor), and Eq. [(30)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-mm-general-reflection)) above.
**2. Phase shift definitions**
The relative phase shift \\(\Delta = \delta_\parallel - \delta_\bot\\) between “\\(\parallel\\)” and “\\(\bot\\)” (Eq. [(19)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-phase-shifts)) is often also presented in opposite form as
(33)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq33 "Link to this equation")\\[\begin{align} \Delta' = \delta_\bot - \delta_\parallel = -\Delta \end{align}\\]
which will swap the signs of the two \\(\sin(\dots)\\) expressions in Eq. [(28)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-mm-tir), Eq. [(29)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-mm-conductor), and Eq. [(30)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-mm-general-reflection).
**3. Phase shift directions**
The phase shifts \\(\delta_\parallel\\) and \\(\delta_\bot\\) are sometimes also interpreted as phase _retardations_ instead of _advances_ , which also influences the same signs, canceling the previous sign convention (Fresnel vs. Verdet convention) again.
### Validation against measurements[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#validation-against-measurements "Link to this heading")
Due to the many possible sources of subtle errors and sign flips in the implementation of the Fresnel equation we also validated the output of Mitsuba 3 against real world measurements from
  1. An accurate in-plane acquisition system built in our lab.
  2. The image-based pBRDF measurements by Baek et al. [[BZK+20](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id22 "Seung-Hwan Baek, Tizian Zeltner, Hyun Jin Ku, Inseung Hwang, Xin Tong, Wenzel Jakob, and Min H. Kim. Image-based acquisition and modeling of polarimetric reflectance. Transactions on Graphics \(Proceedings of SIGGRAPH\), July 2020. doi:10.1145/3386569.3392387.")].


#### In-plane measurements[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#in-plane-measurements "Link to this heading")
At the core of this lies the classical ellipsometry approach of _dual-rotating retarders (DRR)_ proposed by Azzam [[Azz78](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id36 "R. M. A. Azzam. Photopolarimetric measurement of the mueller matrix by fourier analysis of a single detected signal. Opt. Lett., 2\(6\):148–150, Jun 1978. URL: https://ol.osa.org/abstract.cfm?URI=ol-2-6-148, doi:10.1364/OL.2.000148.")]. This technique is remarkably simple and works by only analyzing a single intensity signal that interacted with the material to be measured and a handful of basic optical elements (two linear polarizers and two quarter-wave plates). The setup looks as follows:
[![../../_images/azzam.svg](https://mitsuba.readthedocs.io/en/stable/_images/azzam.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/azzam.svg)
A beam of light passes through a combination of linear polarizer and quarter-wave plate (together referred to as a _polarizer module_), then interacts with the sample to be measured, before passing through another quarter-wave plate and polarizer combination (the _analyzer module_). Both polarizers stay at fixed angles whereas the two retarder elements rotate at speeds 1x and 5x.
Compared to the original description of the setup [[Azz78](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id36 "R. M. A. Azzam. Photopolarimetric measurement of the mueller matrix by fourier analysis of a single detected signal. Opt. Lett., 2\(6\):148–150, Jun 1978. URL: https://ol.osa.org/abstract.cfm?URI=ol-2-6-148, doi:10.1364/OL.2.000148.")] we use a variant which has the two polarizers rotated \\(90˚\\) from each other, i.e. they are at a non-transmissive configuration which is much easier to calibrate compared to the fully transmissive case. For us, it is the first quarter-wave plate that rotates at the faster speed.
The measured signal \\(f(\phi)\\) can therefore be defined as
(34)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq34 "Link to this equation")\\[\begin{equation} f(\phi) = \mathbf{P}(\pi/2) \cdot \mathbf{Q}(\phi) \cdot \mathbf{M} \cdot \mathbf{Q}(5\phi) \cdot \mathbf{P}(0) \end{equation}\\]
for \\(\phi \in [0, \pi]\\), linear polarizers \\(\mathbf{P}(\phi)\\) and quater-wave plates \\(\mathbf{Q}(\phi)\\) at angles \\(\phi\\), and \\(\mathbf{M}\\) the unknown Mueller matrix of the sample that is being observed.
Azzam showed that the coefficients of a 12-term Fourier series expansion
(35)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#eq35 "Link to this equation")\\[\begin{equation} f(\phi) = a_0 + \sum_{k=1}^{12} \big( a_k \cos(2k \phi) + b_k \sin(2k \phi) \big) \end{equation}\\]
can be used to directly infer the 9 Mueller matrix entries \\(\mathbf{M}_{i,j}\\).
To validate this process, we can plug in arbitrary Mueller matrices into this expression and do a “virtual measurement” which will then “reconstruct” the matrix values. As the intensity of the incoming light beam is arbitrary, matrices elements are recovered up to some unknown scale factor. In the following we thus consistently scale all matrices to have \\(\mathbf{M}_{0,0} = 1\\).
**Air** (no sample)
[![../../_images/drr_simulated_signal_air.svg](https://mitsuba.readthedocs.io/en/stable/_images/drr_simulated_signal_air.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/drr_simulated_signal_air.svg) [![../../_images/drr_simulated_matrix_air.svg](https://mitsuba.readthedocs.io/en/stable/_images/drr_simulated_matrix_air.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/drr_simulated_matrix_air.svg)
**Linear polarizer** (horizontal transmission)
[![../../_images/drr_simulated_signal_polarizer.svg](https://mitsuba.readthedocs.io/en/stable/_images/drr_simulated_signal_polarizer.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/drr_simulated_signal_polarizer.svg) [![../../_images/drr_simulated_matrix_polarizer.svg](https://mitsuba.readthedocs.io/en/stable/_images/drr_simulated_matrix_polarizer.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/drr_simulated_matrix_polarizer.svg)
**Quarter-wave plate** (fast axis horizontal)
[![../../_images/drr_simulated_signal_retarder.svg](https://mitsuba.readthedocs.io/en/stable/_images/drr_simulated_signal_retarder.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/drr_simulated_signal_retarder.svg) [![../../_images/drr_simulated_matrix_retarder.svg](https://mitsuba.readthedocs.io/en/stable/_images/drr_simulated_matrix_retarder.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/drr_simulated_matrix_retarder.svg)
* * *
Our physical setup mostly follows the diagram above. The two notable differences are:
  1. Instead of the first linear polarizer we use a polarizing beamsplitter. This again simplifies calibration as it guarantees the linear polarization of the beam to be perfectly aligned with the optical table.
  2. Due to space limitations, we use two \\(45˚\\) mirrors to redirect the laser. This takes place before the _polarizer module_ so it does not affect the measurements.

[![../../_images/azzam_lab_setup.jpg](https://mitsuba.readthedocs.io/en/stable/_images/azzam_lab_setup.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/azzam_lab_setup.jpg)
Both the sample and the analyzer module are mounted such that they can rotate independently of each other. This means, we can probe the same with any combination of incident and outgoing angles \\(\theta_i\\), \\(\theta_o\\). This video showcases the measurement process for a small set of these angles:
As part of the system calibration we first perform a measurement without any sample, effectively capturing the air and any potential inaccuracies of the device. As demonstrated in the following plot, the measured signal follows closely the expected curve from theory and hence the reconstructed Mueller matrix \\(\mathbf{M}^{\text{air}}\\) is very close to the identity:
[![../../_images/drr_calibration.svg](https://mitsuba.readthedocs.io/en/stable/_images/drr_calibration.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/drr_calibration.svg)
  

\\[\begin{split}\begin{equation} \mathbf{M}^{\text{air}} = \begin{bmatrix} 1.00048 & 0.04183 & -0.00323 & -0.00198 \\\ 0.00466 & 1.03467 & -0.00114 & -0.01816 \\\ -0.01167 & 0.00397 & 1.03684 & -0.00046 \\\ 0.00007 & -0.00043 & 0.00172 & 0.99927 \end{bmatrix} \end{equation}\end{split}\\]
For the actual measurements we then use \\(\mathbf{M}^{\text{air}}\\) as a correction term by multiplying its inverse:
\\[\begin{equation} \mathbf{M}^{\text{final}} = (\mathbf{M}^{\text{air}})^{-1} \cdot \mathbf{M} \end{equation}\\]
* * *
We measured two representative materials (conductor and dielectric) that can be accurately described by the Fresnel equations:
  1. An unprotected [gold mirror (M03)](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=8851).
  2. An absorptive [neutral density filter (NG1), made from Schott glass](https://www.thorlabs.com/NewGroupPage9_PF.cfm?Guide=10&Category_ID=220&ObjectGroup_ID=5011). This filter absorbs most of the refracted light and thus is very similar to observing only the reflection component of the Fresnel equations on dielectrics.


In the following we show the DRR signal (densely measured over all \\(\theta_i = -\theta_o\\) configurations of perfect specular reflection) and the reconstructed Mueller matrix entries compared to the analytical version from Mitsuba 3 (plotted over all \\(\theta_i\\) angles). Note that some areas (highlighted in gray) cannot be measured due to the sensor or sample holder blocking the light beam.
For readability, we only show the relevant Mueller matrix entries that are expected to be non-zero for the Fresnel equations:
\\[\begin{split}\begin{equation} \begin{bmatrix} A & B & 0 & 0 \\\ B & A & 0 & 0 \\\ 0 & 0 & C & S1 \\\ 0 & 0 & S2 & C \end{bmatrix} \end{equation}\end{split}\\]
In both the conductor and dielectric case we observe excellent agreement between theory and captured data.
**Gold (conductor)**
[![../../_images/drr_gold_m03_lab.svg](https://mitsuba.readthedocs.io/en/stable/_images/drr_gold_m03_lab.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/drr_gold_m03_lab.svg) [![../../_images/drr_gold_m03_analytic.svg](https://mitsuba.readthedocs.io/en/stable/_images/drr_gold_m03_analytic.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/drr_gold_m03_analytic.svg)
**Schott NG1 (dielectric)**
[![../../_images/drr_schott_ng1_lab.svg](https://mitsuba.readthedocs.io/en/stable/_images/drr_schott_ng1_lab.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/drr_schott_ng1_lab.svg) [![../../_images/drr_schott_ng1_analytic.svg](https://mitsuba.readthedocs.io/en/stable/_images/drr_schott_ng1_analytic.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/drr_schott_ng1_analytic.svg)
#### Image-based measurements[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#image-based-measurements "Link to this heading")
We also compare against the gold measurement available in the [pBRDF database](http://vclab.kaist.ac.kr/siggraph2020/pbrdfdataset/kaistdataset.html) acquired by Baek et al. [[BZK+20](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id22 "Seung-Hwan Baek, Tizian Zeltner, Hyun Jin Ku, Inseung Hwang, Xin Tong, Wenzel Jakob, and Min H. Kim. Image-based acquisition and modeling of polarimetric reflectance. Transactions on Graphics \(Proceedings of SIGGRAPH\), July 2020. doi:10.1145/3386569.3392387.")]. In comparison to our setup that measures only the _in-plane_ \\(\theta_i, \theta_o\\) angles, this data covers the full isotropic pBRDF but with lower accuracy. Nonetheless, the encoded Mueller matrices agree qualitatively with our implementation.
[![../../_images/drr_gold_m03_kaist.svg](https://mitsuba.readthedocs.io/en/stable/_images/drr_gold_m03_kaist.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/drr_gold_m03_kaist.svg) [![../../_images/drr_gold_m03_analytic.svg](https://mitsuba.readthedocs.io/en/stable/_images/drr_gold_m03_analytic.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/drr_gold_m03_analytic.svg)
For details about this capture setup, please refer to the details given in the corresponding article.
### Validation against ART[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#validation-against-art "Link to this heading")
#### Comparison[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#comparison "Link to this heading")
During development of Mitsuba 3 we also compared its polarized output against the existing implementation in the [Advanced Rendering Toolkit (ART) research rendering system](https://cgg.mff.cuni.cz/ART/).
We used a few simple test scenes that we were able to reproduce in both systems that involve specular reflections and refractions. In each case, we directly compare the Stokes vector output (see also Section [Stokes vector output](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#stokes-vector-output)) of the two systems. For ART, this information is normalized (s.t. \\(\mathbf{s}_0 = 1\\)) so we apply this consistently also to our results.
**Dielectric reflection**
A simple Cornell box scene with a dielectric sphere (\\(\eta = 1.5\\)) that causes reflection and refraction.
[![../../_images/art_comparison_dielectric.jpg](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_dielectric.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_dielectric.jpg) [![../../_images/art_comparison_dielectric_s1.jpg](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_dielectric_s1.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_dielectric_s1.jpg) [![../../_images/art_comparison_dielectric_s2.jpg](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_dielectric_s2.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_dielectric_s2.jpg)
**Dielectric internal reflection**
The same scene as before, but the IOR is reversed (\\(\eta = 1/1.5\\)) which causes internal reflection.
[![../../_images/art_comparison_invdielectric.jpg](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_invdielectric.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_invdielectric.jpg) [![../../_images/art_comparison_invdielectric_s1.jpg](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_invdielectric_s1.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_invdielectric_s1.jpg) [![../../_images/art_comparison_invdielectric_s2.jpg](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_invdielectric_s2.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_invdielectric_s2.jpg)
**Conductor reflections**
This scene uses two conductor spheres (\\(\eta = 0.052 - 3.905 i\\)) and also causes elliptical polarization due to the phase shifts and two-bounce reflections between the spheres.
[![../../_images/art_comparison_conductor.jpg](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_conductor.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_conductor.jpg) [![../../_images/art_comparison_conductor_s1.jpg](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_conductor_s1.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_conductor_s1.jpg) [![../../_images/art_comparison_conductor_s2.jpg](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_conductor_s2.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_conductor_s2.jpg) [![../../_images/art_comparison_conductor_s3.jpg](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_conductor_s3.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_conductor_s3.jpg)
#### Bugfix in ART[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#bugfix-in-art "Link to this heading")
During the initial comparison, we discovered a subtle bug in ART’s implementation of the Fresnel equations. In their system, these are implemented based on an alternative formulation [[WW12](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id26 "Alexander Wilkie and Andrea Weidlich. Polarised light in computer graphics. In SIGGRAPH Asia 2012 Courses, SA '12. New York, NY, USA, 2012. Association for Computing Machinery. URL: https://doi.org/10.1145/2407783.2407791, doi:10.1145/2407783.2407791.")], originally from [[WK90](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id44 "L.B. Wolff and D.J. Kurlander. Ray tracing with polarization parameters. IEEE Computer Graphics and Applications, 10\(6\):44-55, 1990. doi:10.1109/38.62695.")]:
\\[\begin{split}\begin{align} F_{\bot} &= \frac{a^{2} + b^{2} - 2 a \cos\theta + \cos^{2}\theta}{a^{2} + b^{2} + 2 a \cos\theta + \cos^{2}\theta} \\\ F_{\parallel} &= \frac{a^{2} + b^{2} - 2 a \sin\theta \tan\theta + \sin^{2}\theta \tan^{2}\theta}{a^{2} + b^{2} + 2 a \sin\theta \tan\theta + \sin^{2}\theta \tan^{2}\theta} F_{\bot} \\\ \tan\delta_{\bot} &= \frac{2 b \cos\theta}{\cos^{2}\theta - a^{2} - b^{2}} \label{eq:art_fresenl_tan_s} \\\ \tan\delta_{\parallel} &= \frac{2 \cos\theta \left[ (n^{2} - k^{2})b - 2 n k a \right]}{(n^{2} + k^{2})^{2} \cos^{2}\theta - a^{2} - b^{2}} \label{eq:art_fresenl_tan_p} \end{align}\end{split}\\]
with
\\[\begin{split}\begin{align} \eta = n + i k \\\ 2 a^{2} &= \sqrt{(n^{2} - k^{2} - \sin^{2}\theta)^{2} + 4 n^{2}k^{2}} + n^{2} - k^{2} - \sin^{2}\theta \\\ 2 b^{2} &= \sqrt{(n^{2} - k^{2} - \sin^{2}\theta)^{2} + 4 n^{2}k^{2}} - n^{2} + k^{2} + \sin^{2}\theta \end{align}\end{split}\\]
The Mueller matrix then follows this shape
\\[\begin{split}\begin{equation} \begin{bmatrix} A & B & 0 & 0 \\\ B & A & 0 & 0 \\\ 0 & 0 & C & S \\\ 0 & 0 & -S & C \end{bmatrix} \end{equation}\end{split}\\]
using
\\[\begin{split}\begin{align} A &= \frac{F_{\bot} + F_{\parallel}}{2} \\\ B &= \frac{F_{\bot} - F_{\parallel}}{2} \\\ C &= \cos(\delta_{\bot} - \delta_{\parallel}) \cdot \sqrt{F_{\bot} \cdot F_{\parallel}} \\\ S &= \sin(\delta_{\bot} - \delta_{\parallel}) \cdot \sqrt{F_{\bot} \cdot F_{\parallel}} \end{align}\end{split}\\]
The phase shifts \\(\delta_{\bot}\\) and \\(\delta_{\parallel}\\) are recovered using the [arctan2](https://en.wikipedia.org/wiki/Atan2) function. Essentially this is similar to our formulation where we take the argument / angle of the complex reflection amplitudes (Eq. [(19)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-phase-shifts)).
Most programming environments (C++, NumPy, MATLAB, …) consistently use the notation \\(\theta = \text{arctan2}(y, x)\\) for computing \\(\arctan(y/x)\\) without ambiguities. However, the ART source code accidentally used a flipped argument order \\(\text{arctan2}(x, y)\\) that is found e.g. in Mathematica.
Such an issue is particularly subtle for the following reasons:
  1. This only affects the sign / handedness of the circular polarization component, which is in most cases of relatively little importance.
  2. Circular polarization only arises when light that is already polarized in some way undergoes an additional phase shift, e.g. by total internal reflection on a dielectric or simple reflection on a conductor. Specifically, a single specular reflection of unpolarized light will never produce circular polarization.


One case where this can be observed is our _Conductor reflections_ test scene from above. We illustrate this sign flip by repeating the comparison of the relevant \\(\mathbf{s}_3\\) Stokes component in that scene — but with the previous flawed implementation.
[![../../_images/art_comparison_conductor_wrong_s3.jpg](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_conductor_wrong_s3.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/art_comparison_conductor_wrong_s3.jpg)
After communication with the authors of ART, this issue will be addressed in the next release.
## Implementation[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#implementation "Link to this heading")
We now turn to the actual implementation of polarized rendering im Mitsuba 3. Due to its [retargetable architecture](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html#sec-variants), the whole system is already built on top of a templated `Spectrum` type and in principle it is very easy to use this mechanism to introduce the required Mueller/Stokes representations but some manual extra effort needs to be made to carefully place the correct Stokes coordinate system rotations. Implementing various forms of Mueller matrices (linear polarizers, retarders, Fresnel equations, …) covered in Section [Mathematics of polarized light](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#mathematics-of-polarized-light) and Section [Fresnel equations](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#fresnel-equations) are also more or less straightforward and they can be found in the corresponding source file `include/mitsuba/render/mueller.h`.
As often is the case however, the devil lies in the details and throughout the development we ran into various issues related to coordinate systems and sign flips. In hindsight, these can all be attributed to mixing conventions from different sources [[Muller69](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id33 "Rolf H. Muller. Definitions and conventions in ellipsometry. Surface Science, 16:14-33, August 1969. doi:10.1016/0039-6028\(69\)90003-X.")] or misconceptions about what they mean.
We briefly list these here for reference and to help people avoid these issues in the future:
  1. We initially found it natural to track the Stokes reference frames (Section [Reference frames](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#reference-frames)) from the viewpoint of the light source. After a lot of confusion about the commonly used rotation matrices (Section [Rotation of Stokes vector frames](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rotation-of-stokes-vector-frames)) and the meaning of left/right handed circular polarization we realized that almost all optics sources defines this reference frame the other way around, using the point of view of the sensor.
  2. The “\\(\bot\\)” and “\\(\parallel\\)” components of the transverse wave (Section [Theory](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#theory)) used in the Fresnel equations (i.e. perpendicular or parallel to the plane of incidence) are pretty clearly defined but we initially did not correctly interpret how these map to the Mueller-Stokes calculus. In particular, we (wrongly) assumed that the “\\(\parallel\\)” direction should be the “horizontal” axis of the incident and outgoing Stokes vectors. We discovered this issue after the measurements of a few representative materials (Section [Validation against measurements](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#validation-against-measurements)) which could not be explained in any other way. The fact that “\\(\bot\\)” corresponds to the “horizontal” axis is also clear when taking a closer look at how the Fresnel equations are actually converted into Mueller matrices [[Col71](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id29 "Edward Collett. Mueller-stokes matrix formulation of fresnel's equations. American Journal of Physics, 39\(5\):517-528, 1971. doi:10.1119/1.1986205.")]. Surprisingly, this discussion is missing in many sources that include these matrices.
  3. We were surprised to find that the complex index of refraction of conductors is most commonly defined with a negative imaginary component, i.e. \\(\eta = n - k i\\) (Section [Equations](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equations)). In contrast, the usual definition in computer graphics uses a positive sign (\\(\eta = n + k i\\)) [[PJH16](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id31 "Matt Pharr, Wenzel Jakob, and Greg Humphreys. Physically Based Rendering: From Theory to Implementation \(3rd ed.\). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 3rd edition, October 2016. ISBN 9780128006450.")]. (In fact, both signs produce the same outcome if we assume no polarized light.) This is also a point where our measurements (Section [Validation against measurements](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#validation-against-measurements)) were really useful. As the Fresnel equations need to behave consistently between dielectrics (\\(k = 0\\)) and conductors (\\(k > 0\\)), another good validation was to make sure there is no sudden sign flip between a dielectric (e.g. \\(\eta = 1.5\\)) and a conductor with only a tiny amount of extinction (e.g. \\(\eta = 1.5 - 0.0001 i\\)).
  4. One of the biggest confusions we faced is related to the Fresnel equations themselves (Section [Different conventions in the literature](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#different-conventions-in-the-literature) and Section [Different conventions in the context of Mueller matrices](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#different-conventions-in-the-context-of-mueller-matrices)). For instance, the _Fresnel_ or _Verdet conventions_ of defining the underlying electric fields cause subtle sign flips in some of the equations. You can also define phase shifts (e.g. on total internal reflection) as either phase _retardations_ or _advances_. Again, this only flips a few signs but together this gives a few different combinations of signs in the resulting Mueller matrices (e.g. Eq. [(30)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-mm-general-reflection)) — and you can find all of them in various sources. All of them are correct (in their respective conventions) but this situation makes it very hard to work with multiple references simultaneously.


We originally used _Stellar polarimetry_ [[Cla09](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id28 "David Clarke. Stellar Polarimetry, Appendix A: The Fresnel Laws. John Wiley & Sons, Ltd, 2009. ISBN 9783527628322. doi:https://doi.org/10.1002/9783527628322.app1.")] as our main sources so our implementation was consistent from the beginning. However the formulation there uses the Fresnel convention and phase delays and we did not initially understand why most other sources we looked had things written down differently. (And we always came back to this question whenever some other part of the implementation was not behaving as expected.) Towards the end of development (when we had consistency with the measurements) and after a thorough literature review we decided to switch Mitsuba 3 to the much more common Verdet convention and phase advances.
The remainder of this document serves as an overview of all relevant parts of the code base and it will hopefully be useful for both understanding and extending the polarization specific aspects of Mitsuba 3.
### Representation[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#representation "Link to this heading")
#### Stokes and Mueller types[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#stokes-and-mueller-types "Link to this heading")
Using the Mueller-Stokes calculus (Section [Mathematics of polarized light](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#mathematics-of-polarized-light)) in a renderer introduces a type mismatch between fundamental quantities used in different parts of the system. For instance, the `Spectrum` array type (with either a single monochromatic entry, 3 RGB values, or intensities at sampled wavelengths) would normally be used to represent emission, reflectance, and importance in non-polarized renderers. Polarization requires us to change it to Stokes vectors in the former case, and Mueller matrices in the latter two cases however.
We avoid this asymmetry we made the decision to only use Mueller matrices where in the case of Stokes vectors, only the first column is non-zero:
[![../../_images/stokes_vector_matrix.svg](https://mitsuba.readthedocs.io/en/stable/_images/stokes_vector_matrix.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/stokes_vector_matrix.svg)
This leads to some unnecessary arithmetic but simplifies the API which is especially useful when implementing bidirectional techniques. In particular, Mitsuba 3 has a single API (`include/mitsuba/render/endpoint.h`) that is shared by both emitters and sensors.
* * *
When also considering the different [color representations in Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html#sec-variants-colors) there are three possible Mueller matrix types:
[![../../_images/color_modes.svg](https://mitsuba.readthedocs.io/en/stable/_images/color_modes.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/color_modes.svg)
1. **Monochrome mode** : \\(4 \times 4 \times 1\\) matrices.
  

This mode completely disables all concept of color in Mitsuba 3, and instead, only the intensity / luminance of light is simulated.
  

Example: `scalar_mono_polarized`.
2. **RGB mode** : \\(4 \times 4 \times 3\\) matrices
  

RGB colors are used in many rendering systems for its simplicity. However it can be a [poor approximation of how color actually works](https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html#sec-variants-colors). The accuracy is even more questionable in polarized rendering modes, as e.g. the Fresnel equations for conductors will not be evaluated at wavelengths with actual physical meaning.
  

Example: `scalar_rgb_polarized`.
3. **Spectral mode** : \\(4 \times 4 \times 4\\) matrices
  

A full spectral color representation is the recommended approach to use for polarized rendering. The last dimension is \\(4\\) because Mitsuba 3 by default traces four randomly sampled wavelengths at once.
  

Example: `scalar_spectral_polarized`.
* * *
The remainder of this section covers a few more implementation details / helpful functions involving the `Spectrum` type and polarization.
  1. There is a type trait to detect whether the `Spectrum` type is polarized or not that can be used for code blocks that are only needed in polarized variants. This is especially helpful in conjunction with the C++17 `constexpr` feature:


```
1// From `include/mitsuba/core/traits.h`
2template<typenameT>constexprboolis_polarized_v=...
3
4// Example use case
5ifconstexpr(is_polarized_v<Spectrum>){
6// ... only compiled in polarized modes ...
7}

```
Copy to clipboard
2. There is a similar type trait to turn a polarized `Spectrum` (i.e. a matrix) into an unpolarized form (i.e. only a 1D/3D/4D vector depending on the underlying color representation).
This is useful for instance when querying reflectance data from a bitmap texture where no polarization specific information is stored.
```
1// From `include/mitsuba/core/traits.h`
2template<typenameT>usingunpolarized_spectrum_t=...
3
4// Example use case
5usingUnpolarizedSpectrum=unpolarized_spectrum_t<Spectrum>;

```
Copy to clipboard
3. There exists a helper function to extract the \\((1, 1)\\) entry of the Mueller matrix / Stokes vector. Essentially, it will turn any `Spectrum` value into an `UnpolarizedSpectrum`.
This is used in a few places in the renderer where we do not care about the additional polarization information tracked by the Mueller matrix. For instance, when performing Russian Roulette (stochastic termination of long light paths) based on the current path throughput or when writing a final RGB pixel value to the image, see Section [Rendering output](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rendering-output).
```
 1// From `include/mitsuba/core/spectrum.h`
 2
 3template<typenameT>
 4unpolarized_spectrum_t<T>unpolarized_spectrum(constT&s){
 5ifconstexpr(is_polarized_v<T>){
 6// First entry of the Mueller matrix is the unpolarized spectrum
 7returns(0,0);
 8}else{
 9returns;
10}
11}

```
Copy to clipboard
4. Another helper function returns a depolarizing Mueller matrix (with only its \\((1, 1)\\) entry used) where the input is usually an `UnpolarizedSpectrum` value.
This is obviously used in places where we want materials to act as depolarizers, e.g. in the case of a Lambertian diffuse material. However, there are also many BSDFs where it is currently not clear how they should interact with polarized light, or the functionality is not yet implemented. For now, these all act as depolarizers. Lastly, this is also used for light sources, as they currently all emit completely unpolarized light in Mitsuba 3.
```
 1// From `include/mitsuba/core/spectrum.h`
 2
 3template<typenameT>
 4autodepolarizer(constT&s=T(1)){
 5ifconstexpr(is_polarized_v<T>){
 6Tresult=dr::zero<T>();
 7result(0,0)=s(0,0);
 8returnresult;
 9}else{
10returns;
11}
12}
13
14// Example use case, where the value returned from the texture is of type
15// `UnpolarizedSpectrum`.
16Spectrums=depolarizer<Spectrum>(m_texture->eval(si,active));

```
Copy to clipboard
#### Coordinate frames[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#coordinate-frames "Link to this heading")
A second complication of the Mueller-Stokes calculus is the need to keep track of reference frames throughout the rendering process. One choice (e.g. used by the [ART rendering system](https://cgg.mff.cuni.cz/ART/)) would be to store incident and outgoing frames with each Mueller matrix type directly (where one of the two is redundant in case the matrix encodes a Stokes vector).
To better leverage the retargetable design of Mitsuba 3 where the `Spectrum` template type is substituted for arrays of varying dimensions we chose to instead represent these coordinate systems implicitly. For each unit vector, we can construct its orthogonal Stokes basis vector:
_Constructing a unique Stokes basis vector for the given unit vector._[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id70 "Link to this code")
```
1// From `include/mitsuba/render/mueller.h`
2template<typenameVector3>
3Vector3stokes_basis(constVector3&omega){
4returncoordinate_system(omega).first;
5}

```
Copy to clipboard
Here, the `coordinate_system` function constructs two orthogonal vectors to `omega` in a deterministic way [[DBC+17](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id34 "Tom Duff, James Burgess, Per Christensen, Christophe Hery, Andrew Kensler, Max Liani, and Ryusuke Villemin. Building an orthonormal basis, revisited. Journal of Computer Graphics Techniques \(JCGT\), 6\(1\):1–8, March 2017. URL: https://jcgt.org/published/0006/01/01/.")]. The resulting Stokes basis vector is therefore always unique for a given direction.
Recall from Section [Mueller matrices](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#mueller-matrices) that whenever two Mueller matrices (or a Mueller matrix and a Stokes vector) are multiplied with each other, their respective outgoing and incident reference frames need to be aligned with each other. This is usually achieved by some combination of rotator matrices (Eq. [(3)](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#equation-eq-rotator-matrix)) that transform the Stokes frames (Section [Rotation of Stokes vector frames](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rotation-of-stokes-vector-frames)). Mitsuba 3 implements two versions where the rotation angle \\(\theta\\) is either directly known, or inferred from a provided set of basis vectors before and after the desired rotation:
_Rotate Stokes vector reference frames._[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id71 "Link to this code")
```
 1// From `include/mitsuba/render/mueller.h`
 2
 3// Construct the Mueller matrix that rotates the reference frame of a Stokes
 4// vector by an angle `theta`.
 5template<typenameFloat>
 6MuellerMatrix<Float>rotator(Floattheta){
 7auto[s,c]=dr::sincos(2.f*theta);
 8returnMuellerMatrix<Float>(
 91,0,0,0,
100,c,s,0,
110,-s,c,0,
120,0,0,1
13);
14}
15
16// Construct the Mueller matrix that rotates the reference frame of a Stokes
17// vector from `basis_current` to `basis_target`.
18// The masked condition ensures that rotation is performed in the right direction.
19template<typenameVector3,
20typenameFloat=dr::value_t<Vector3>,
21typenameMuellerMatrix=MuellerMatrix<Float>>
22MuellerMatrixrotate_stokes_basis(constVector3&forward,
23constVector3&basis_current,
24constVector3&basis_target){
25Floattheta=unit_angle(dr::normalize(basis_current),
26dr::normalize(basis_target));
27
28autoflip=dr::dot(forward,dr::cross(basis_current,basis_target))<0;
29dr::masked(theta,flip)*=-1.f;
30returnrotator(theta);
31}

```
Copy to clipboard
We then have more functions that build on top of this to cover the common cases of rotating incident/outgoing Mueller reference frames (**left** , see Section [Rotation of Mueller matrix frames](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rotation-of-mueller-matrix-frames)) and rotated optical elements (**right** , see Section [Rotation of optical elements](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rotation-of-optical-elements)):
[![../../_images/rotated_mueller_matrix_crop.png](https://mitsuba.readthedocs.io/en/stable/_images/rotated_mueller_matrix_crop.png) ](https://mitsuba.readthedocs.io/en/stable/_images/rotated_mueller_matrix_crop.png) [![../../_images/rotated_element_crop.png](https://mitsuba.readthedocs.io/en/stable/_images/rotated_element_crop.png) ](https://mitsuba.readthedocs.io/en/stable/_images/rotated_element_crop.png)
_Rotate Mueller matrix reference frames._[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id72 "Link to this code")
```
 1// From `include/mitsuba/render/mueller.h`
 2
 3// "Rotate" a Mueller matrix in the general case (i.e. when both
 4// sides use a different basis).
 5//
 6// Before:
 7//     Matrix M operates from `in_basis_current` to `out_basis_current` frames.
 8// After:
 9//     Returned matrix operates from `in_basis_target` to `out_basis_target` frames.
10template<typenameVector3,
11typenameFloat=dr::value_t<Vector3>,
12typenameMuellerMatrix=MuellerMatrix<Float>>
13MuellerMatrixrotate_mueller_basis(constMuellerMatrix&M,
14constVector3&in_forward,
15constVector3&in_basis_current,
16constVector3&in_basis_target,
17constVector3&out_forward,
18constVector3&out_basis_current,
19constVector3&out_basis_target){
20MuellerMatrixR_in=rotate_stokes_basis(in_forward,
21in_basis_current,
22in_basis_target);
23MuellerMatrixR_out=rotate_stokes_basis(out_forward,
24out_basis_current,
25out_basis_target);
26returnR_out*M*transpose(R_in);
27}
28
29// Special case of `rotate_mueller_basis`.
30// "Rotate" a Mueller matrix in the case of collinear incident and
31// outgoing directions (i.e. when both sides have the same basis).
32//
33// Before:
34//     Matrix M operates from `basis_current` to `basis_current` frames.
35// After:
36//     Returned matrix operates from `basis_target` to `basis_target` frames.
37template<typenameVector3,
38typenameFloat=dr::value_t<Vector3>,
39typenameMuellerMatrix=MuellerMatrix<Float>>
40MuellerMatrixrotate_mueller_basis_collinear(constMuellerMatrix&M,
41constVector3&forward,
42constVector3&basis_current,
43constVector3&basis_target){
44MuellerMatrixR=rotate_stokes_basis(forward,basis_current,basis_target);
45returnR*M*transpose(R);
46}
47
48// Apply a rotation to the Mueller matrix of a given optical element.
49template<typenameFloat>
50MuellerMatrix<Float>rotated_element(Floattheta,
51constMuellerMatrix<Float>&M){
52MuellerMatrix<Float>R=rotator(theta),Rt=transpose(R);
53returnRt*M*R;
54}

```
Copy to clipboard
Both of these require a `forward` unit vector that points along the propagation direction of light.
### Algorithms and materials[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#algorithms-and-materials "Link to this heading")
With the discussion about representation of light via Stokes vectors and scattering via Mueller matrices out of the way we can now turn to the more rendering specific aspects of the implementation: the algorithms, material models, and how the two are coupled.
#### Handling of light paths[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#handling-of-light-paths "Link to this heading")
In principle, not much changes when adding polarization to a rendering algorithm and the retargetable type system of Mitsuba 3 will do a lot of the heavy lifting. In particular, emitters will return emission in form of Stokes vectors (or rather Mueller matrices with three zero valued columns.) and _polarized bidirectional scattering distribution functions (pBSDFs)_ will return Mueller matrices that are multiplied with the emission:
[![../../_images/light_path_ordering.svg](https://mitsuba.readthedocs.io/en/stable/_images/light_path_ordering.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/light_path_ordering.svg)
As it turns out, polarization breaks some of the symmetry of light transport that is usually exploited by rendering algorithms [[MSWK16](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id4 "Michal Mojzik, Tomas Skrivan, Alexander Wilkie, and Jaroslav Krivanek. Bi-Directional Polarised Light Transport. In Eurographics Symposium on Rendering. 2016.")], [[JA18](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id3 "Adrian Jarabo and Victor Arellano. Bidirectional rendering of vector light transport. Computer Graphics Forum, 2018.")], especially for bidirectional techniques. As a consequence, the actual direction of light propagation (i.e. from emitter to sensor) needs to be kept in mind at all times as it dictates the ordering in which matrices will be multiplied.
* * *
Consider the two extreme ends of bidirectional algorithms that construct a light path with \\(n\\) vertices \\((\mathbf{x}_0, \dots, \mathbf{x}_{n-1}\\)) where the ordering is from emitter to sensor as in the figure above:
1. **Light tracing** follows the natural ordering of events: we sample rays starting from light sources and their emitted light (in form of a Stokes vector \\(\mathbf{s}_0\\)) is multiplied by Mueller matrices \\(\mathbf{M}_k\\) returned from pBSDFs encountered along all scattering events \\(k\\) of a light path.
  

During path generation, a _throughput_ variable \\(\boldsymbol{\beta}_k^{\text{(lt)}}\\) is updated which is the sequence of previous Mueller matrices multiplied onto the original emission vector from the left:
\\[\boldsymbol{\beta}_k^{\text{(lt)}} = \mathbf{M}_k \cdot \mathbf{M}_{k-1} \cdot \dots \cdot \mathbf{M}_1 \cdot \mathbf{s}_0\\]
2. **Path tracing** , on the other hand, follows things in reverse: rays are sampled starting from the sensor side. As conceptually the same computation needs to take place as in the previous case, the path _throughput_ at each bounce is now a Mueller matrix \\(\mathbf{B}_k^{\text{(pt)}}\\) where Mueller matrices from encountered pBSDFs are multiplied from the right:
\\[\mathbf{B}_k^{\text{(pt)}} = \mathbf{M}_{n-1} \cdot \mathbf{M}_{n-2} \cdot \dots \cdot \mathbf{M}_k\\]
Only when hitting the light source (vertex \\(k=0\\)) directly, or connecting to it via _next event estimation_ will the Stokes vector \\(\mathbf{s}_0\\) be actually multiplied as well.
* * *
Polarized rendering algorithms compute Stokes vectors as output where either only their intensity (1st entry) or all individual components are written into some image format, see Section [Rendering output](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rendering-output) below.
#### pBSDF evaluation and sampling[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#pbsdf-evaluation-and-sampling "Link to this heading")
During Section [Handling of light paths](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#handling-of-light-paths) we glossed over the tricky question of coordinate system rotations and silently assumed that all Mueller matrix multiplications are taking place in their valid reference frames. Mitsuba 3 deals with this issue in a consistent way that is tightly coupled to how BSDF evaluations and sampling take place.
**Default case without polarization**
For completeness, we will first briefly discuss the usual convention **without polarization** specific changes. Also see `include/mitsuba/render/bsdf.h` for details about the BSDF API.
Mitsuba 3 (like most modern rendering systems) uses BSDF implementations that are completely decoupled from the actual rendering algorithms in order to be as extensible as possible. To facilitate this, their sampling and evaluation is taking place in some canonical _local_ coordinate system that is always aligned with the shading normal \\(\mathbf{n}\\) pointing _up_ (towards \\(+\mathbf{z}\\)):
[![../../_images/world_local_unpol.svg](https://mitsuba.readthedocs.io/en/stable/_images/world_local_unpol.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/world_local_unpol.svg)
Note also how both incident and outgoing directions \\(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o\\) point “away” from the center and that, due to reciprocity of BSDFs, they are completely interchangeable Care only needs to be taken when sampling the BSDFs, i.e. when only one direction is given initially, and a new one should be generated. In this scenario, Mitsuba 3 (arbitrarily) has the convention that \\(\boldsymbol{\omega}_i\\) is given, and \\(\boldsymbol{\omega}_o\\) is newly sampled.
As rendering algorithms take place in _world space_ however, some transformations between the two spaces are necessary:
_``to_local`` and ``to_world`` space transformations._[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id73 "Link to this code")
```
 1// From `include/mitsuba/core/frame.h`
 2
 3// Convert from world coordinates to local coordinates
 4// with orthogonal frame `(s, t, n)`.
 5Vector3fto_local(constVector3f&v)const{
 6returnVector3f(dr::dot(v,s),dr::dot(v,t),dr::dot(v,n));
 7}
 8
 9// Convert from local coordinates with orthogonal
10// frame `(s, t, n)` to world coordinates.
11Vector3fto_world(constVector3f&v)const{
12returns*v.x()+t*v.y()+n*v.z();
13}

```
Copy to clipboard
**Extra considerations with polarization**
In a setting **with polarization** this aspect becomes more involved as there are now also Stokes basis vectors associated with the local & world space incident and outgoing directions that are subject to these transformations:
[![../../_images/world_local_pol.svg](https://mitsuba.readthedocs.io/en/stable/_images/world_local_pol.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/world_local_pol.svg)
Mitsuba 3 makes heavy use of its _implicit_ Stokes coordinate frames (Section [Coordinate frames](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#coordinate-frames)) here. Mueller matrices returned from pBSDF evaluation or sampling are always assumed to be valid for the implicit bases of the local unit vectors (\\(-\boldsymbol{\omega}_o^{\text{local}}\\) and \\(\boldsymbol{\omega}_i^{\text{local}}\\) in the Figure above). Or outlined more clearly in code:
```
 1BSDFContextctx;// Used to pass optional flags to BSDFs
 2SurfaceInteraction3fsi=...;// Returned from ray-scene intersections.
 3
 4// Incident and outgoing directions in world space
 5Vector3fwo=...;
 6Vector3fwi=...;
 7
 8// Transform into local space ...
 9Vector3fwo_local=si.to_local(wo);
10si.wi=si.to_local(wi);
11
12// ... and evaluate the pBSDF.
13Spectrumbsdf_val=bsdf->eval(ctx,si,wo_local);
14
15// The returned Mueller matrix `bsdf_val` is then valid for the
16// following input & output Stokes reference bases that are defined
17// implicitly.
18Vector3fbo_local=stokes_basis(-wo_local);
19Vector3fbi_local=stokes_basis(si.wi);

```
Copy to clipboard
In order to perform the correct rotations, throughout this description the two unit vectors need to point along the propagation direction of light. In practice, this means that compared to the unpolarized Figure above, the vector pointing towards the emitter side needs to be reversed.
At this point we have all the information to transform the Mueller matrix such that it is valid in the global Stokes bases `bo = stokes_basis(-wo)` and `bi = stokes_basis(wi)` where it can be safely multiplied as discussed in Section [Handling of light paths](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#handling-of-light-paths) earlier. This involves converting these basis into a common space and applying a Mueller matrix rotation (Section [Coordinate frames](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#coordinate-frames)). All of this is done as part of a single helper function:
_``to_world_mueller`` space transformations for Mueller matrices._[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id74 "Link to this code")
```
 1// From `include/mitsuba/render/interaction.h`
 2
 3// Transform a Mueller matrix `M_local` that is the result of evaluating or sampling
 4// a pBSDF in local space to world space.
 5// `in_forward_local` and `out_forward_local` are the unit vectors (also in local
 6// space) that point along the light propagation direction before and after the
 7// scattering event.
 8Spectrumto_world_mueller(constSpectrum&M_local,
 9constVector3f&in_forward_local,
10constVector3f&out_forward_local)const{
11ifconstexpr(is_polarized_v<Spectrum>){
12// Incident and outgoing directions are transformed into world space
13Vector3fin_forward_world=to_world(in_forward_local),
14out_forward_world=to_world(out_forward_local);
15
16// Establish the "current" and "target" incident basis:
17//     "current": The implicit basis of the local-space incident
18direction,transformedintoworldspace.
19//     "target":  The implicit basis of the world-space incident direction.
20Vector3fin_basis_current=to_world(mueller::stokes_basis(in_forward_local)),
21in_basis_target=mueller::stokes_basis(in_forward_world);
22
23// Analogously establish the same for the outgoing direction.
24Vector3fout_basis_current=to_world(mueller::stokes_basis(out_forward_local)),
25out_basis_target=mueller::stokes_basis(out_forward_world);
26
27// Rotate the Mueller matrix frames to transform correctly between the two.
28returnmueller::rotate_mueller_basis(M_local,
29in_forward_world,in_basis_current,in_basis_target,
30out_forward_world,out_basis_current,out_basis_target);
31}else{
32// No-op in unpolarized variants.
33returnM_local;
34}

```
Copy to clipboard
The complementary `to_local_mueller` function is also implemented but is usually not needed apart from debugging purposes.
Finally, the following two code snippets show heavily commented pBSDF evaluation and sampling steps that include all of the considerations above. This is comparable to what is done e.g. in the current path tracer implementation of Mitsuba 3 (`src/integrators/path.cpp`). Note that only the lines involving `to_world_mueller` had to be added compared to a standard unpolarized path tracer.
_Evaluate a pBSDF_[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id75 "Link to this code")
```
 1BSDFContextctx;// Used to pass optional flags to BSDFs
 2SurfaceInteraction3fsi=...;// Returned from ray-scene intersections.
 3
 4// Incident and outgoing directions in world space
 5Vector3fwo=...;
 6Vector3fwi=...;
 7
 8// Transform into local space ...
 9Vector3fwo_local=si.to_local(wo);
10si.wi=si.to_local(wi);
11
12// ... and evaluate the BSDF
13Spectrumbsdf_val=bsdf->eval(ctx,si,wo_local);
14
15// At this point, the Mueller matrix `bsdf_val` is valid in the implicit
16// reference frames of local space vectors `-wo_local` and `si.wi`.
17// Both unit vectors follow the direction of the light. (We assume an
18// algorithm similar to path tracing here. When tracing from the emitter side,
19// things will be reversed.)
20
21// Transform Mueller matrix into world space
22bsdf_val=si.to_world_mueller(bsdf_val,-wo_local,si.wi);
23
24// Now, the Mueller matrix `bsdf_val` is valid in the implicit
25// reference frames of world space vectors `-wo` and `wi`.

```
Copy to clipboard
_Sample from a pBSDF_[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id76 "Link to this code")
```
 1BSDFContextctx;
 2SurfaceInteraction3fsi=...;
 3
 4// Incident direction in world space ...
 5Vector3fwi=...;
 6// ... and transformed into local space
 7si.wi=si.to_local(wi);
 8
 9// Sample based on random numbers (in local space)
10auto[bs,bsdf_weight]=bsdf->sample(ctx,sampler.next_1d(),sampler.next_2d());
11
12// At this point, the Mueller matrix `bsdf_weight` is valid in the implicit
13// reference frames of local space vectors `-bs.wo` and `si.wi`.
14// Both unit vectors follow the direction of the light. (We assume an
15// algorithm similar to path tracing here. When tracing from the emitter side,
16// things will be reversed.)
17
18// Transform sampled direction into world space
19Vector3fwo=si.to_world(bs.wo);
20
21// Transform Mueller matrix into world space
22bsdf_weight=si.to_world_mueller(bsdf_weight,-bs.wo,si.wi);
23
24// Now, the Mueller matrix `bsdf_weight` is valid in the implicit
25// reference frames of world space vectors `-wo` and `wi`.

```
Copy to clipboard
#### Implementing pBSDFs[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#implementing-pbsdfs "Link to this heading")
After looking at rendering algorithms and how they interact with pBSDFs, we still need to discuss the internals of a given pBSDF implementation, i.e. what is happening in the (local space) evaluation and sampling routines.
As mentioned above, these should always return valid Mueller matrices based on the implicit reference frames of the (local) incident and outgoing directions. In the following Figure, these are marked as \\(\mathbf{b}_o^{\text{local}}\\) and \\(\mathbf{b}_i^{\text{local}}\\).
[![../../_images/bsdf_coord_change.svg](https://mitsuba.readthedocs.io/en/stable/_images/bsdf_coord_change.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/bsdf_coord_change.svg)
In terms of code, they are computed as
```
1Vector3fbo_local=stokes_basis(-wo_local);
2Vector3fbi_local=stokes_basis(wi_local);

```
Copy to clipboard
When pBSDFs construct Mueller matrices, these are usually based on some very specific Stokes basis convention, e.g. described in textbooks. For instance, the current pBSDFs found in Mitsuba 3 are mostly built from Mueller matrices based on the Fresnel equations. This means, they are constructed based on the perpendicular (”\\(\bot\\)”) and parallel (”\\(\parallel\\)”) vectors relative to the plane of reflection. Please refer back to Section [Fresnel equations](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#fresnel-equations) for details. In particular, the main “horizontal” or \\(\mathbf{b}_{i/o}\\) vector is the “\\(\bot\\)” component that can be constructed as follows for incident and outgoing directions:
```
1Vector3fn(0,0,1);// Surface normal
2Vector3fs_axis_in=dr::normalize(dr::cross(n,-wo_local)),
3s_axis_out=dr::normalize(dr::cross(n,wi_local));

```
Copy to clipboard
As illustrated in the Figure above, a (final) coordinate rotation needs to take place to convert the Mueller matrix to the right bases before it can be returned from the pBSDF.
For this rotation, we need to know along what direction the light is traveling so we need to correctly distinguish between \\(\boldsymbol{\omega}_i\\) and \\(\boldsymbol{\omega}_o\\). During path tracing (or any algorithm that traces from the sensor side), Mitsuba 3 uses BSDFs with the convention that light arrives from \\(-\boldsymbol{\omega}_o\\) and leaves along \\(\boldsymbol{\omega}_i\\) (because \\(\boldsymbol{\omega}_o\\) is sampled based on \\(\boldsymbol{\omega}_i\\) that points towards the sensor side). During light tracing from emitters, the roles are reversed however: light arrives from \\(-\boldsymbol{\omega}_i\\) and leaves along \\(\boldsymbol{\omega}_o\\). Mitsuba 3 has a special BSDF flag (`TransportMode`) that carries the relevant information to resolve this confusion and is used for similar “non-reciprocal” situations in bi-directional techniques [[Vea98](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id25 "Eric Veach. Robust Monte Carlo Methods for Light Transport Simulation. Stanford University, Stanford, CA, USA, 1998. ISBN 0591907801.")].
Putting everything together, here is a short snippet as example of how the a pBSDF evaluation based on the Fresnel equations can be implemented. This is very close to what is done for instance in `src/bsdfs/conductor.cpp`.
Example pBSDF evaluation based on Fresnel equations.[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id77 "Link to this code")
```
 1// Incident and outgoing directions, given in local space.
 2Vector3fwo_local=...;
 3Vector3fwi_local=...;
 4
 5// Due to the coordinate system rotations for polarization-aware
 6// pBSDFs below we need to know the propagation direction of light.
 7// In the following, light arrives along `-wo_hat` and leaves along
 8// `+wi_hat`.
 9//
10// `TransportMode` has two states:
11//     - `Radiance`, meaning we trace from the sensor to the light sources
12//     - `Importance`, meaning we trace from the light sources to the sensor
13//
14Vector3fwo_hat=ctx.mode==TransportMode::Radiance?wo_local:wi_local,
15wi_hat=ctx.mode==TransportMode::Radiance?wi_local:wo_local;
16
17// Querty the complex index of refraction
18dr::Complex<UnpolarizedSpectrum>eta=...;
19
20// Evaluate the Mueller matrix for specular reflection.
21// This will internally call the Fresnel equations and assemble the correct
22// matrix.
23// The incident angle cosine is cast to `UnpolarizedSpectrum` to be compatible with
24// `eta` when calling the function.
25UnpolarizedSpectrumcos_theta=Frame3f::cos_theta(wo_hat);
26Spectrumvalue=mueller::specular_reflection(cos_theta,eta);
27
28// Apply the "handedness change" Mueller matrix that is necessary for reflections.
29// Recall from the Section about Fresnel equations that this is a diagonal matrix
30// with entries [1, 1, -1, -1].
31value=mueller::reverse(value);
32
33// Compute the Stokes reference frame vectors of this matrix that are
34// perpendicular to the plane of reflection.
35Vector3fn(0,0,1);
36Vector3fs_axis_in=dr::normalize(dr::cross(n,-wo_hat)),
37s_axis_out=dr::normalize(dr::cross(n,wi_hat));
38
39// Rotate in/out reference vector of `value` s.t. it aligns with the implicit
40// Stokes bases of -wo_hat & wi_hat. */
41value=mueller::rotate_mueller_basis(value,
42-wo_hat,s_axis_in,mueller::stokes_basis(-wo_hat),
43wi_hat,s_axis_out,mueller::stokes_basis(wi_hat));

```
Copy to clipboard
### Rendering output[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#rendering-output "Link to this heading")
In this section we will briefly discuss the output side of the rendering system and how to extract polarization specific information from it to use in experiments.
#### Intensity output[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#intensity-output "Link to this heading")
In most natural scenes, polarization effects are very subtle and hardly visible to the eye. Enabling polarized rendering modes in Mitsuba 3 therefore also usually does not produce vastly different outputs compared to an unpolarized mode. However, in particular cases such as specular interreflections or as soon as polarizing optical elements (such as linear polarizers) are introduced to the scene differences can be noticed.
Here we compare unpolarized (**left**) vs polarized (**right**) renderings of a simple Cornell box scene with both dielectric and conductor materials and a microfacet based rough conductor pattern on the otherwise diffuse walls. Because differences are still subtle in this case, we also show a visualization of the pixel-wise absolute error.
[![../../_images/scene_unpol.png](https://mitsuba.readthedocs.io/en/stable/_images/scene_unpol.png) ](https://mitsuba.readthedocs.io/en/stable/_images/scene_unpol.png)
Spectral rendering mode[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id78 "Link to this image")
[![../../_images/scene_pol.png](https://mitsuba.readthedocs.io/en/stable/_images/scene_pol.png) ](https://mitsuba.readthedocs.io/en/stable/_images/scene_pol.png)
Spectral + Polarized rendering mode[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id79 "Link to this image")
[![../../_images/scene_diff.jpg](https://mitsuba.readthedocs.io/en/stable/_images/scene_diff.jpg) ](https://mitsuba.readthedocs.io/en/stable/_images/scene_diff.jpg)
Pixel-wise absolute error between the two previous images.[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id80 "Link to this image")
Alternatively, the difference is clearer when opening both images in separate tabs and switching between them.
Another simple but effective way to make the effects visible is to place a (rotating) linear polarizer in front of the camera. Such animations clearly show the underlying complexity that arises from accounting for polarization:
#### Stokes vector output[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#stokes-vector-output "Link to this heading")
Mitsuba 3 can optionally also output the complete Stokes vector output at the end of the simulation by writing a multichannel EXR image. This is accomplished by switching to a special [Stokes integrator plugin](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#integrator-stokes) when setting up the scene description `.xml` file:
```
1<integratortype="stokes">
2<!-- Note how there is still a normal path tracer nested inside that
3         will do the actual simulation. -->
4<integratortype="path"/>
5</integrator>

```
Copy to clipboard
The output EXR images produced by this encodes the Stokes vectors with 16 channels in total:
  * (**0-3**): The normal color (RGB) + alpha (A) channels.
  * (**4-6**): \\(\mathbf{s}_0\\) (intensity) as an RGB image.
  * (**7-9**): \\(\mathbf{s}_1\\) (horizontal vs. vertical polarization) as an RGB image.
  * (**10-12**): \\(\mathbf{s}_2\\) (diagonal linear polarization) as an RGB image.
  * (**13-15**): \\(\mathbf{s}_3\\) (right vs. left circular polarization) as an RGB image.


Currently, the Stokes components (\\(\mathbf{s}_0, \dots, \mathbf{s}_3\\)) will all go through the usual color conversion process at the end. In spectral mode, this can be problematic as Stokes vectors for different wavelengths will be integrated against XYZ sensitivity curves and finally converted to RGB colors (while still being signed quantities). This is likely not ideal for all applications, so alternatively we suggest to render images at fixed wavelengths. This can be achieved for instance by constructing scenes carefully s.t. only uniform spectra are used everywhere. Monochromatic or RGB rendering modes can be a useful tool here as these use “raw” floating point inputs and outputs. In comparison, spectral modes use an upsampling step to determine plausible spectra based on input RGB values which might not work as expected.
Internally, the Stokes integrator has to perform a last coordinate frame rotation to make sure the Stokes vector is saved in a consistent format, where its reference basis is aligned with the horizontal axis of the camera frame:
[![../../_images/camera_rotation.svg](https://mitsuba.readthedocs.io/en/stable/_images/camera_rotation.svg) ](https://mitsuba.readthedocs.io/en/stable/_images/camera_rotation.svg)
Or in source code:
The final alignment with the computed Stokes vector and the horizontal axis of the camera.[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id81 "Link to this code")
```
 1// From `src/integrators/stokes.cpp`
 2
 3// Call a nested integrator (e.g. the path tracer)
 4auto[result,mask]=integrator->sample(scene,sampler,ray,...);
 5
 6// Compute the implicit Stokes reference for the incoming light path
 7Vector3fbasis_out=mueller::stokes_basis(-ray.d);
 8
 9// Get the camera transformation and evaluate for the current sampled `time`
10Transform4ftransform=scene->sensors()[0]->world_transform()->eval(ray.time);
11
12// Compute the output Stokes reference that aligns horizontally with the camera
13Vector3fvertical=transform*Vector3f(0.f,1.f,0.f);
14Vector3fbasis_cam=dr::cross(ray.d,vertical);
15
16// Perform the final Mueller matrix reference frame rotation on the output
17result=mueller::rotate_stokes_basis(-ray.d,basis_out,basis_cam)*result;
18
19// Extract the first column of the Mueller matrix, i.e. the Stokes vector
20autoconst&stokes=result.entry(0);

```
Copy to clipboard
#### Polarization visualizations[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#polarization-visualizations "Link to this heading")
Mitsuba 3 also includes a separate command line tool to create commonly used visualizations from the EXR images with Stokes vector information, which can be run via Python, e.g.:
```
python-mmitsuba.python.polvis<filename>.exr<flags>

```
Copy to clipboard
For instance, it can create the following false-color visualizations (green: positive, red: negative) of the Stokes components of the scene above:
[![../../_images/scene_s0.png](https://mitsuba.readthedocs.io/en/stable/_images/scene_s0.png) ](https://mitsuba.readthedocs.io/en/stable/_images/scene_s0.png)
\\(\mathbf{s}_0\\)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id82 "Link to this image")
[![../../_images/scene_s1.png](https://mitsuba.readthedocs.io/en/stable/_images/scene_s1.png) ](https://mitsuba.readthedocs.io/en/stable/_images/scene_s1.png)
\\(\mathbf{s}_1\\)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id83 "Link to this image")
[![../../_images/scene_s2.png](https://mitsuba.readthedocs.io/en/stable/_images/scene_s2.png) ](https://mitsuba.readthedocs.io/en/stable/_images/scene_s2.png)
\\(\mathbf{s}_2\\)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id84 "Link to this image")
[![../../_images/scene_s3.png](https://mitsuba.readthedocs.io/en/stable/_images/scene_s3.png) ](https://mitsuba.readthedocs.io/en/stable/_images/scene_s3.png)
\\(\mathbf{s}_3\\)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id85 "Link to this image")
Alternatives that are more intuitive, originally proposed by Wilkie and Weidlich [[WW10](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id43 "Alexander Wilkie and Andrea Weidlich. A standardised polarisation visualisation for images. In Proceedings of the 26th Spring Conference on Computer Graphics, SCCG '10, 43–50. New York, NY, USA, 2010. Association for Computing Machinery. URL: https://doi.org/10.1145/1925059.1925070, doi:10.1145/1925059.1925070.")], are also supported. Please refer to the corresponding article for more information on these.
[![../../_images/scene_dop.png](https://mitsuba.readthedocs.io/en/stable/_images/scene_dop.png) ](https://mitsuba.readthedocs.io/en/stable/_images/scene_dop.png)
Degree of polarization[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id86 "Link to this image")
[![../../_images/scene_top.png](https://mitsuba.readthedocs.io/en/stable/_images/scene_top.png) ](https://mitsuba.readthedocs.io/en/stable/_images/scene_top.png)
Type of polarization (linear vs. circular)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id87 "Link to this image")
[![../../_images/scene_lin.png](https://mitsuba.readthedocs.io/en/stable/_images/scene_lin.png) ](https://mitsuba.readthedocs.io/en/stable/_images/scene_lin.png)
Orientation of linear polarization[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id88 "Link to this image")
[![../../_images/scene_cir.png](https://mitsuba.readthedocs.io/en/stable/_images/scene_cir.png) ](https://mitsuba.readthedocs.io/en/stable/_images/scene_cir.png)
Chirality of circular polarization (left vs. right)[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id89 "Link to this image")
#### Choosing an integrator[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#choosing-an-integrator "Link to this heading")
Like for regular rendering, the normal [path tracer](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#integrator-path) is a great default choice for polarized rendering. But it does have one notable limitation that tends to come up in polarized scenes, especially ones that use optical elements (e.g. polarizers, retarders) for instance when using Mitsuba 3 to build a virtual version of optical experiments.
Consider the following (seemingly) simple Cornell box scene, where a relatively small light source is behind a linear polarizer:
[![../../_images/scene_nee_ref.png](https://mitsuba.readthedocs.io/en/stable/_images/scene_nee_ref.png) ](https://mitsuba.readthedocs.io/en/stable/_images/scene_nee_ref.png)
In such scenes, a path tracer will normally rely heavily on the ability to connect _shadow rays_ directly to the light source to account for the incoming illumination at a shading point. Unfortunately, the path tracer implemented in Mitsuba 3 is not able to do so through the linear polarizer and as a consequence, it has to rely on hitting the light source purely by chance after diffuse scattering on one of the walls. Depending on the size of the light source, this can be a source of very high variance that will take a long time to converge, see the image below (**left**).
Because light travels through the polarizer in a straight line (unlike for instance when rendering caustics through some glass objects), these _shadow rays_ can in principle be traced through the filter. In such scenarios, it is recommended to switch to the [volumetric path tracer](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#integrator-volpath) that does have support for this (see image below, **right**) even if the scene does not actually contain any volumes.
Both of these images were rendered in roughly equal time at the same number of samples per pixel. The only difference is the choice of the integrator. Note also that both images are correct _in expectation_ , and the path tracer will just need a much longer time to converge to the solution.
[![../../_images/scene_nee_path.png](https://mitsuba.readthedocs.io/en/stable/_images/scene_nee_path.png) ](https://mitsuba.readthedocs.io/en/stable/_images/scene_nee_path.png)
Rendered with the path tracer[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id90 "Link to this image")
[![../../_images/scene_nee_volpath.png](https://mitsuba.readthedocs.io/en/stable/_images/scene_nee_volpath.png) ](https://mitsuba.readthedocs.io/en/stable/_images/scene_nee_volpath.png)
Rendered with the volumetric path tracer[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#id91 "Link to this image")
### Feature list[¶](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#feature-list "Link to this heading")
The following plugins in Mitsuba 3 currently work/interact with polarization:
**Integrators**
  * Direct illumination ([direct](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#integrator-direct))
  * Path tracer ([path](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#integrator-path))
  * Volumetric path tracer when used for surfaces ([volpath](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#integrator-volpath))
  * Stokes integrator to extract polarization specific information ([stokes](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html#integrator-stokes))


**BSDFs**
  * Smooth conductor ([conductor](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-conductor))
  * Smooth dielectric ([dielectric](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-dielectric))
  * Rough conductor ([roughconductor](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-roughconductor))
  * Linear polarizer ([polarizer](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-polarizer))
  * Linear retarder ([retarder](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-retarder))
  * Circular polarizer ([circular](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-circular))
  * Null ([null](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-null))
  * Polarized plastic from [[BJTK18](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id23 "Seung-Hwan Baek, Daniel S. Jeon, Xin Tong, and Min H. Kim. Simultaneous acquisition of polarimetric svbrdf and normals. ACM Transactions on Graphics \(Proc. SIGGRAPH Asia 2018\), 37\(6\):268:1–15, 2018. URL: https://dx.doi.org/10.1145/3272127.3275018, doi:10.1145/3272127.3275018.")] ([pplastic](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-pplastic))
  * Measured polarized to render pBRDFs measured as part of [[BZK+20](https://mitsuba.readthedocs.io/en/stable/zz_bibliography.html#id22 "Seung-Hwan Baek, Tizian Zeltner, Hyun Jin Ku, Inseung Hwang, Xin Tong, Wenzel Jakob, and Min H. Kim. Image-based acquisition and modeling of polarimetric reflectance. Transactions on Graphics \(Proceedings of SIGGRAPH\), July 2020. doi:10.1145/3386569.3392387.")] ([measured_polarized](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-measured-polarized))


Polarized BSDFs can also be combined or modified with these plugins:
  * Blended material ([blendbsdf](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-blendbsdf))
  * Bump map adapter ([bumpmap](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-bumpmap))
  * Normal map adapter ([normalmap](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-normalmap))
  * Opacity mask ([mask](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-mask))
  * Two-sided adapter ([twosided](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#bsdf-twosided))


All remaining BSDFs currently act as depolarizers.
* * *
At this point, the polarization support in Mitsuba 3 has two main functionalities that are missing: polarized emission and volumetric scattering.
**Polarized emission**
All emitters in Mitsuba 3 currently emit completely unpolarized light. This can be partially sidestepped for now by placing filters (e.g. linear polarizer) into the scene. A more complete solution would require careful Stokes reference basis conversions between emitter coordinate systems and world space, similarly as for the BSDF case (Section [pBSDF evaluation and sampling](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#pbsdf-evaluation-and-sampling)).
**Polarized volumetric scattering**
While the volumetric path tracer (`volpath`) does support polarization when restricted to the surface case, it does not support polarized volumes. In particular, all phase functions in Mitsuba 3 act as depolarizers currently. Support for this would likely require small adjustments to the integrator (for tracking coordinate systems) and implementing some polarized phase function. A lot of the coordinate conversions from the BSDF case could probably be omitted as phase functions do not have a concept of _local space_ and always operate in _world space_ (Section [pBSDF evaluation and sampling](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html#pbsdf-evaluation-and-sampling)).
* * *
