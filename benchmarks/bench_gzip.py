"""Opening a gzipped MRC file."""

import _harness as h

import mrcfile


def main():
    path = h.fixture("gz_src.mrc.gz")
    h.warm(path)

    def open_gz():
        with mrcfile.open(path) as m:
            return m.data[0, 0, 0]

    h.emit("gzip", {"open_gz_64MiB": h.measure(open_gz, repeats=2)})


main()
