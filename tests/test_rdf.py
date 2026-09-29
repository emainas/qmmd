from pathlib import Path

from qmmd.rdf import _trajin_line, load_config, write_cpptraj_in


def test_rdf_frame_override_and_named_provenance(tmp_path: Path) -> None:
    config_path = tmp_path / "rdf.yaml"
    config_path.write_text(
        """
system: APP
buffer: 4.0
method_dir: dftb
bench_tag: N1T64C1
run_ids: "1-10"
cv_dir: equil
traj_name: traject
dftb_inp_name: dftb.inp
parm_path: ready.parm7
analysis_name: deprot-na-water-h
frames:
  start: 1
  stride: 1
  stop_by_run:
    8: 2001
rdf:
  dr: 0.01
  r_max: 7.0
  mask1: "@1"
  mask2: ":WAT@H*"
  volume: true
"""
    )
    cfg = load_config(config_path)

    assert _trajin_line(cfg, Path("traject"), 7) == "trajin traject"
    assert _trajin_line(cfg, Path("traject"), 8) == "trajin traject 1 2001 1"

    cpptraj_input = write_cpptraj_in(
        cfg, Path("ready.parm7"), Path("traject"), tmp_path, 8
    )
    assert cpptraj_input.name == "cpptraj_deprot_na_water_h.in"
    text = cpptraj_input.read_text()
    assert "trajin traject 1 2001 1" in text
    assert "radial out rdf_1_WAT_H.dat 0.01 7.0 @1 :WAT@H* volume" in text
