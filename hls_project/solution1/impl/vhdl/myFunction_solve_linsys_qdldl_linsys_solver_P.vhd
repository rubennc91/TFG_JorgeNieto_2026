-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity myFunction_solve_linsys_qdldl_linsys_solver_P_rom is 
    generic(
             DWIDTH     : integer := 6; 
             AWIDTH     : integer := 6; 
             MEM_SIZE    : integer := 34
    ); 
    port (
          addr0      : in std_logic_vector(AWIDTH-1 downto 0); 
          ce0       : in std_logic; 
          q0         : out std_logic_vector(DWIDTH-1 downto 0);
          clk       : in std_logic
    ); 
end entity; 


architecture rtl of myFunction_solve_linsys_qdldl_linsys_solver_P_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "011100", 1 => "011011", 2 => "001111", 3 => "000000", 4 => "000110", 
    5 => "011101", 6 => "010101", 7 => "000011", 8 => "010010", 9 => "100001", 
    10 => "011001", 11 => "001010", 12 => "011000", 13 => "011111", 14 => "001001", 
    15 => "011110", 16 => "011010", 17 => "010000", 18 => "000001", 19 => "010001", 
    20 => "000010", 21 => "000111", 22 => "000100", 23 => "001000", 24 => "010111", 
    25 => "010110", 26 => "000101", 27 => "010011", 28 => "010100", 29 => "001100", 
    30 => "001011", 31 => "001101", 32 => "001110", 33 => "100000" );


begin 


memory_access_guard_0: process (addr0) 
begin
      addr0_tmp <= addr0;
--synthesis translate_off
      if (CONV_INTEGER(addr0) > mem_size-1) then
           addr0_tmp <= (others => '0');
      else 
           addr0_tmp <= addr0;
      end if;
--synthesis translate_on
end process;

p_rom_access: process (clk)  
begin 
    if (clk'event and clk = '1') then
        if (ce0 = '1') then 
            q0 <= mem(CONV_INTEGER(addr0_tmp)); 
        end if;
    end if;
end process;

end rtl;

Library IEEE;
use IEEE.std_logic_1164.all;

entity myFunction_solve_linsys_qdldl_linsys_solver_P is
    generic (
        DataWidth : INTEGER := 6;
        AddressRange : INTEGER := 34;
        AddressWidth : INTEGER := 6);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of myFunction_solve_linsys_qdldl_linsys_solver_P is
    component myFunction_solve_linsys_qdldl_linsys_solver_P_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    myFunction_solve_linsys_qdldl_linsys_solver_P_rom_U :  component myFunction_solve_linsys_qdldl_linsys_solver_P_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


