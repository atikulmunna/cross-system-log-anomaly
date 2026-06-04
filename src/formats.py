"""Per-system LogHub log formats and header->regex conversion."""
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
