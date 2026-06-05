# Roadmap

This roadmap summarizes the planned development of `rough-contact-etm`. The project is currently a research-oriented workflow for thermo-electro-mechanical simulation of rough surface contact. The roadmap is not a strict schedule; it is intended to make the development direction transparent and to help users understand which physical models and numerical features are planned.

## v0.2.x — Research preview

The current version focuses on a clean and reproducible baseline implementation.

Main goals:

* Elastic rough-surface contact using Tamaas.
* FFT-based electrical and thermal Green's-function solvers.
* Joule-heating calculation from the local current density.
* Thermoelastic surface displacement feedback.
* HDF5-based output for load-dependent simulations.
* Minimal examples and basic tests.
* Documentation of model assumptions, limitations, and workflow.

## v0.3.x — Current-carrying frictional contact

The next development stage will extend the model from normal contact to frictional current-carrying contact.

Planned features:

* Tangential loading and frictional slip/stick state variables.
* Coupled frictional heating and Joule heating.
* Current-density redistribution under partial slip.
* Contact-state-dependent electrical conductance.
* Example cases for comparing purely electrical heating, purely frictional heating, and combined heating.
* Post-processing tools for slip maps, frictional power density, current-density localization, and temperature hot spots.

Scientific motivation:

This extension is intended to support studies of electrified sliding and rolling contacts where mechanical friction, electrical current flow, and local heating interact strongly.

## v0.4.x — Elastoplastic rough contact

This stage will introduce plasticity-related corrections or reduced-order elastoplastic models for rough contact.

Planned features:

* Elastoplastic asperity-level correction models.
* Pressure or hardness-limited contact response.
* Load-dependent real contact area with elastic-to-plastic transition.
* Electrical constriction resistance models affected by plastic contact spots.
* Comparison with elastic-only predictions.
* Benchmark examples for rough surfaces with different RMS roughness, spectral bandwidth, and material hardness.

Scientific motivation:

Purely elastic contact may overestimate local pressure and underestimate the evolution of real contact area at high loads. Elastoplastic extensions are important for more realistic electrical and thermal contact predictions.

## v0.5.x — Electrical breakdown and discharge modeling

This stage will add local breakdown and discharge-related modeling.

Planned features:

* Local electric-field or voltage-drop breakdown criteria.
* Active-set representation of discharged regions.
* Post-discharge voltage clamping or local conductance enhancement.
* Energy-release estimates for discharge events.
* Coupling between discharge sites, current localization, and thermal hot spots.
* Example cases showing the transition from isolated discharge spots to connected discharge bands.

Scientific motivation:

Electrical breakdown is relevant to electrified bearings, lubricated contacts, and rough interfaces subjected to high voltage or high current density. The goal is to provide a transparent numerical framework for studying where discharge is likely to occur and how it interacts with contact geometry.

## v0.6.x — Lubricated and EHL-related extensions

This stage will connect the rough-contact ETM workflow with lubricated contact models.

Planned features:

* Thin-film electrical resistance and capacitance models.
* Mixed contact representation with solid contact and lubricating film regions.
* Reduced-order EHL film-thickness models.
* Optional coupling to EHL solvers or externally supplied film-thickness fields.
* Load-speed-viscosity parameter sweeps.
* Examples relevant to electrified rolling bearings and lubricated machine elements.

Scientific motivation:

Many engineering contacts operate in mixed or elastohydrodynamic lubrication regimes. Adding film-dependent electrical and thermal pathways will make the framework more useful for bearing electrification and impedance-based diagnostics.

## v1.0.0 — Validated reproducible research workflow

The first stable release will focus on validation, documentation, and reproducibility.

Planned features:

* Reproducible benchmark cases.
* Clear validation against analytical, numerical, or literature reference solutions.
* Stable input configuration format.
* Continuous integration tests for core numerical kernels.
* Improved documentation and tutorials.
* Versioned example outputs.
* Clear API boundaries for contact, electrical, thermal, and post-processing modules.

Long-term vision:

The long-term goal of `rough-contact-etm` is to provide an open, extensible, and reproducible research code for studying thermo-electro-mechanical coupling in rough mechanical contacts, with applications to electrical contacts, electrified bearings, tribology, Joule heating, discharge damage, and condition monitoring.
