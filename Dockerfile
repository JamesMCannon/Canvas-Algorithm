FROM python:3.7

WORKDIR /usr/src/app

# Requirements for pyglow

RUN pip install numpy --upgrade
RUN pip install -r payload/pyglow/requirements.txt
RUN cd payload/pyglow
RUN make -C src/pyglow/models source
RUN python3 setup.py install --user



ENTRYPOINT ["/bin/bash"]