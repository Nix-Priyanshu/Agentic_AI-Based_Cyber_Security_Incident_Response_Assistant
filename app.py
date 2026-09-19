"""
CyberGuardian Connect - Phase 1: Connections

Users link a data source (a REST API web app or an uploaded report), test it,
sync it, and see the incidents that arrive in the common incident format.
The full dashboard, AI analysis and reports are built on top of this in
the next phases.

Run:  streamlit run app.py
"""
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from connectors import ConnectorError, build_connector

st.set_page_config(page_title="CyberGuardian Connect", page_icon="🛡️", layout="wide")

SEVERITY_ICONS = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🔵", "Unknown": "⚪"}


# ---------------------------------------------------------------- state
def init_state():
    st.session_state.setdefault("connections", [])   # saved connection settings
    st.session_state.setdefault("incidents", {})     # connection id -> list of incident dicts
    st.session_state.setdefault("flash", [])         # messages to show after a rerun


def flash(message: str):
    st.session_state.flash.append(message)


def show_flash():
    for message in st.session_state.flash:
        st.success(message)
    st.session_state.flash = []


# ---------------------------------------------------------------- actions
def sync_connection(conn: dict) -> int:
    """Fetch incidents through the connector and keep them in the session."""
    incidents = build_connector(conn).fetch_incidents()
    st.session_state.incidents[conn["id"]] = [incident.to_dict() for incident in incidents]
    conn["last_sync"] = datetime.now()
    return len(incidents)


def save_and_sync(conn: dict):
    st.session_state.connections.append(conn)
    count = sync_connection(conn)
    flash(f"Connection '{conn['name']}' saved. {count} incident(s) synced.")


def remove_connection(conn: dict):
    st.session_state.connections = [c for c in st.session_state.connections if c["id"] != conn["id"]]
    st.session_state.incidents.pop(conn["id"], None)


# ---------------------------------------------------------------- add-connection forms
def check_connection(conn: dict) -> bool:
    """Test the connection, show the result, and return True when it works."""
    try:
        ok, message = build_connector(conn).test_connection()
    except ConnectorError as exc:
        ok, message = False, str(exc)
    (st.success if ok else st.error)(message)
    return ok


def render_rest_form():
    with st.form("add_rest"):
        name = st.text_input("Name", value="Simulated SOC Platform")
        base_url = st.text_input("Base URL", value="http://localhost:8000")
        api_key = st.text_input("API key", type="password", help="Sent as the X-API-Key header.")
        test_col, save_col = st.columns(2)
        test = test_col.form_submit_button("Test connection")
        save = save_col.form_submit_button("Save and sync", type="primary")

    if not (test or save):
        return
    if not (name.strip() and base_url.strip() and api_key.strip()):
        st.error("Enter a name, the base URL and the API key.")
        return
    conn = {"id": uuid.uuid4().hex[:8], "type": "rest", "name": name.strip(),
            "base_url": base_url.strip(), "api_key": api_key.strip(), "last_sync": None}
    if check_connection(conn) and save:
        try:
            save_and_sync(conn)
        except ConnectorError as exc:
            st.session_state.connections = [c for c in st.session_state.connections if c["id"] != conn["id"]]
            st.error(str(exc))
            return
        st.rerun()


def render_file_form():
    with st.form("add_file"):
        name = st.text_input("Name", value="Uploaded report")
        uploaded = st.file_uploader("Incident report (.txt)", type=["txt"])
        save = st.form_submit_button("Save and sync", type="primary")

    if not save:
        return
    if not name.strip() or uploaded is None:
        st.error("Enter a name and choose a .txt report.")
        return
    conn = {"id": uuid.uuid4().hex[:8], "type": "file", "name": name.strip(),
            "filename": uploaded.name,
            "text": uploaded.getvalue().decode("utf-8", errors="replace"), "last_sync": None}
    if check_connection(conn):
        try:
            save_and_sync(conn)
        except ConnectorError as exc:
            st.session_state.connections = [c for c in st.session_state.connections if c["id"] != conn["id"]]
            st.error(str(exc))
            return
        st.rerun()


def render_add_connection():
    with st.expander("＋ Add a connection", expanded=not st.session_state.connections):
        kind = st.radio("Source type", ["REST API", "File upload"], horizontal=True)
        if kind == "REST API":
            render_rest_form()
        else:
            render_file_form()


# ---------------------------------------------------------------- connection cards
def render_connection_card(conn: dict):
    incident_count = len(st.session_state.incidents.get(conn["id"], []))
    with st.container(border=True):
        info_col, button_col = st.columns([3, 2])
        with info_col:
            st.markdown(f"**{conn['name']}**")
            if conn["type"] == "rest":
                st.caption(f"REST API at {conn['base_url']}")
            else:
                st.caption(f"File: {conn['filename']}")
            synced = conn["last_sync"].strftime("%H:%M:%S") if conn["last_sync"] else "never"
            st.caption(f"{incident_count} incident(s), last synced {synced}")
        with button_col:
            test_btn, sync_btn, remove_btn = st.columns(3)
            if test_btn.button("Test", key=f"test_{conn['id']}"):
                check_connection(conn)
            if sync_btn.button("Sync", key=f"sync_{conn['id']}"):
                try:
                    count = sync_connection(conn)
                    flash(f"'{conn['name']}' synced. {count} incident(s).")
                    st.rerun()
                except ConnectorError as exc:
                    st.error(str(exc))
            if remove_btn.button("Remove", key=f"remove_{conn['id']}"):
                remove_connection(conn)
                st.rerun()


# ---------------------------------------------------------------- incident preview
def key_value_frame(values: dict) -> pd.DataFrame:
    rows = []
    for key, value in values.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        rows.append({"Field": key.replace("_", " ").capitalize(), "Value": value or "-"})
    return pd.DataFrame(rows)


def render_incident_preview():
    incidents = [i for conn in st.session_state.connections
                 for i in st.session_state.incidents.get(conn["id"], [])]
    if not incidents:
        st.info("No incidents yet. Add a connection above and sync it.")
        return

    table = pd.DataFrame([{
        "ID": i["id"], "Title": i["title"],
        "Severity": f"{SEVERITY_ICONS.get(i['severity'], '⚪')} {i['severity']}",
        "Status": i["status"], "Created": i["created_at"], "Source": i["source"],
    } for i in incidents])
    st.dataframe(table, hide_index=True)

    chosen = st.selectbox(
        "Inspect an incident",
        options=range(len(incidents)),
        format_func=lambda n: f"{incidents[n]['id']}  {incidents[n]['title'][:70]}",
    )
    incident = incidents[chosen]

    metric_severity, metric_status, metric_actor = st.columns(3)
    metric_severity.metric("Severity", incident["severity"])
    metric_status.metric("Status", incident["status"] or "-")
    metric_actor.metric("Threat actor", incident["threat_actor"] or "Unknown")
    st.write(incident["summary"])

    tab_asset, tab_ioc, tab_mitre, tab_time = st.tabs(
        ["Asset and network", "Indicators", "ATT&CK techniques", "Timeline"]
    )
    with tab_asset:
        left, right = st.columns(2)
        left.markdown("**Affected asset**")
        left.dataframe(key_value_frame(incident["host"]), hide_index=True)
        right.markdown("**Network**")
        right.dataframe(key_value_frame(incident["network"]), hide_index=True)
    with tab_ioc:
        found = False
        for label, values in incident["indicators"].items():
            if values:
                found = True
                st.markdown(f"**{label.capitalize()}** ({len(values)})")
                st.code("\n".join(values), language=None)
        if not found:
            st.caption("No indicators in this incident.")
    with tab_mitre:
        if incident["techniques"]:
            st.dataframe(pd.DataFrame(incident["techniques"]).rename(columns={"id": "ID", "name": "Technique"}),
                         hide_index=True)
        else:
            st.caption("No ATT&CK techniques in this incident.")
    with tab_time:
        if incident["timeline"]:
            st.dataframe(pd.DataFrame(incident["timeline"]).rename(columns={"time": "Time", "event": "Event"}),
                         hide_index=True)
        else:
            st.caption("No timeline in this incident.")


# ---------------------------------------------------------------- page
init_state()

with st.sidebar:
    st.markdown("## 🛡️ CyberGuardian Connect")
    st.caption("Connect your security tools, then analyze and respond.")
    st.divider()
    st.metric("Connections", len(st.session_state.connections))
    st.metric("Incidents synced", sum(len(v) for v in st.session_state.incidents.values()))
    st.divider()
    st.caption("Phase 1: connections. The dashboard, AI analysis and reports come next.")

st.title("Connections")
st.caption("Link a web application or upload a report. Incoming data is converted to one common format.")

show_flash()
render_add_connection()

if st.session_state.connections:
    st.subheader("Your connections")
    for connection in st.session_state.connections:
        render_connection_card(connection)

st.subheader("Incidents")
render_incident_preview()
