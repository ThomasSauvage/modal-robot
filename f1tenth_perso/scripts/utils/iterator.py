def middle_range(minmum: int, maximum: int):
    """Iterate over [|minmum, maximum|] by starting at the middle of the range and then going to the edges."""

    center = (minmum + maximum) // 2
    mid_size = (maximum - minmum) // 2
    offset = 0  # Offset from the center

    while offset <= mid_size:
        yield center + offset
        if offset != 0:
            yield center - offset

        offset += 1


if __name__ == "__main__":
    for i in middle_range(-10, 10):
        print(i)
