import re
from .logger import get_logger
import zmq

# ASTRA-Sim's per-iteration report, the one line of its stdout the frontend
# has to parse. Compiled once: parse_output runs on every handshake, and at
# 8 NPUs a 10-request run makes 337,786 of them.
_ITERATION_RE = re.compile(
    r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, "
    r"exposed communication (\d+) cycles."
)


class Controller():
    def __init__(self, total_num, zmq_addr):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(zmq_addr)

        self.end_dict = {}
        self.total_num = total_num
        self.logger = get_logger(self.__class__)
        for i in range(total_num):
            self.end_dict[i] = -1


    def read_wait(self):
        """Read ASTRA-Sim's stdout up to the "Waiting" prompt.

        Every line before the prompt is the iteration report; ASTRA-Sim used
        to interleave a per-tick "Checking ..." line per NPU, which made this
        loop 3.07M reads on a 10-request 8-NPU run against 675k now. See the
        ASTRA_SIM_TRACE_POLLING note in the analytical backend's main.cc.
        """
        msg = self.socket.recv_string().split(" ")
        assert msg[0] == "Waiting"

        sys = int(msg[1])
        id = int(msg[2])
        cycle = int(msg[3])
        com_cycle = int(msg[4])

        if self.end_dict[sys] != id:
            self.logger.info(
                "NPU[%d] iteration %d finished, %d cycles, exposed communication %d cycles.",
                sys,
                id,
                cycle,
                com_cycle,
            )
            self.end_dict[sys] = id
        return {'sys': sys, 'id': id, 'cycle': cycle}


    def check_end(self, p):
        if (exit_code := p.wait()) != 0:
            raise RuntimeError(f"ASTRA-Sim process has exited with non-zero exit code {exit_code}!")

    def write_cmd(self, input):
        self.socket.send_string(input)