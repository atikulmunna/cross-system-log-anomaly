"""Per-system LogHub log formats and header->regex conversion.

Each format string describes the fixed header fields of a system's log lines,
with <Field> placeholders. The trailing <Content> is the free-text message that
Drain3 turns into a template. Format strings follow the logparser conventions
(regex-significant chars like [] () are pre-escaped in the strings below).
"""
import re

# Canonical logparser format strings for the 16 LogHub systems.
LOG_FORMATS = {
    "HDFS":        "<Date> <Time> <Pid> <Level> <Component>: <Content>",
    "Hadoop":      "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
    "Spark":       "<Date> <Time> <Level> <Component>: <Content>",
    "Zookeeper":   "<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
    "BGL":         "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
    "HPC":         "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
    "Thunderbird": "<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>",
    "Windows":     "<Date> <Time>, <Level>                  <Component>    <Content>",
    "Linux":       "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
    "Android":     "<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>",
    "Apache":      "\[<Time>\] \[<Level>\] <Content>",
    "OpenSSH":     "<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>",
    "OpenStack":   "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>",
    "Mac":         "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
    "HealthApp":   "<Time>\|<Component>\|<Pid>\|<Content>",
    "Proxifier":   "\[<Time>\] <Program> - <Content>",
}

# Systems whose first field is an alert label ('-' == normal, else anomaly).
LABEL_SYSTEMS = {"BGL", "Thunderbird"}


def generate_logformat_regex(logformat):
    """Convert a logparser format string into (headers, compiled_regex)."""
    headers = []
    splitters = re.split(r"(<[^<>]+>)", logformat)
    regex = ""
    for k, part in enumerate(splitters):
        if k % 2 == 0:
            # literal between fields: collapse whitespace to flexible match
            regex += re.sub(r"\s+", r"\s+", part)
        else:
            header = part.strip("<>")
            regex += "(?P<%s>.*?)" % header
            headers.append(header)
    return headers, re.compile("^" + regex + "$")
