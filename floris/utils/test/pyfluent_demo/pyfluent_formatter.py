import ansys.fluent.core as pyfluent

solver_session = pyfluent.launch_fluent(mode="solver")
solver_session.check_health()