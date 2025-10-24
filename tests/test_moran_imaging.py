def test_import():
    from moran_imaging import __version__

    assert isinstance(__version__, str)
