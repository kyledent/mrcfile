"""Full read of an uncompressed MRC file into memory (page cache warm)."""

import _harness as h

import mrcfile


def main():
    path = h.fixture("vol_f32.mrc")
    h.warm(path)

    def read():
        with mrcfile.open(path) as m:
            return m.data[0, 0, 0]

    def read_header_only():
        with mrcfile.open(path, header_only=True) as m:
            return int(m.header.nx)

    h.emit(
        "read",
        {
            "full_read_128MiB": h.measure(read),
            "header_only": h.measure(read_header_only, repeats=20),
        },
    )


main()
