//! Per-component AIR definitions (stwo-cairo-style layout): one file per component holding its
//! `FrameworkEval` and the relation id(s) it owns. `gate` is the demand side (imports the three
//! supply relation ids); `program`/`qubitmem`/`range_check` are the supply tables. The `Components`
//! assembly + shared `LookupElements`/consts stay in `crate::air`.

pub(crate) mod gate;
pub(crate) mod program;
pub(crate) mod qubitmem;
pub(crate) mod range_check;
