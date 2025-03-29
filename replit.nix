{pkgs}: {
  deps = [
    pkgs.sqlite
    pkgs.libGLU
    pkgs.libGL
    pkgs.postgresql
    pkgs.openssl
  ];
}
