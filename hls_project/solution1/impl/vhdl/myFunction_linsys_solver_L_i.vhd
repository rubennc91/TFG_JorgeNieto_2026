-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
--
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity myFunction_linsys_solver_L_i_ram is 
    generic(
            DWIDTH     : integer := 6; 
            AWIDTH     : integer := 6; 
            MEM_SIZE    : integer := 57
    ); 
    port (
          addr0     : in std_logic_vector(AWIDTH-1 downto 0); 
          ce0       : in std_logic; 
          d0        : in std_logic_vector(DWIDTH-1 downto 0); 
          we0       : in std_logic; 
          q0        : out std_logic_vector(DWIDTH-1 downto 0);
          clk        : in std_logic 
    ); 
end entity; 


architecture rtl of myFunction_linsys_solver_L_i_ram is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
shared variable ram : mem_array := (
    0 => "011111", 1 => "100000", 2 => "011110", 3 => "000011", 4 => "001000", 
    5 => "000110", 6 => "011111", 7 => "000111", 8 => "011111", 9 => "001000", 
    10 => "011111", 11 => "011110", 12 => "011111", 13 => "011110", 14 => "011111", 
    15 => "001011", 16 => "001111", 17 to 18=> "001110", 19 => "011110", 20 => "001111", 
    21 => "011110", 22 => "011101", 23 => "011110", 24 => "011101", 25 => "011110", 
    26 => "010010", 27 => "011011", 28 => "010100", 29 => "011011", 30 => "011100", 
    31 to 32=> "011001", 33 => "011011", 34 => "011000", 35 => "011010", 36 => "100000", 
    37 => "011010", 38 => "011011", 39 => "100000", 40 => "011011", 41 => "011100", 
    42 => "100000", 43 => "011100", 44 => "011101", 45 => "100000", 46 => "011101", 
    47 => "100000", 48 => "011110", 49 => "100000", 50 => "100001", 51 => "011111", 
    52 => "100000", 53 => "100001", 54 => "100000", 55 to 56=> "100001" );


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

p_memory_access_0: process (clk)  
begin 
    if (clk'event and clk = '1') then
        if (ce0 = '1') then 
            q0 <= ram(CONV_INTEGER(addr0_tmp));
            if (we0 = '1') then 
                ram(CONV_INTEGER(addr0_tmp)) := d0; 
            end if;
        end if;
    end if;
end process;


end rtl;

Library IEEE;
use IEEE.std_logic_1164.all;

entity myFunction_linsys_solver_L_i is
    generic (
        DataWidth : INTEGER := 6;
        AddressRange : INTEGER := 57;
        AddressWidth : INTEGER := 6);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        we0 : IN STD_LOGIC;
        d0 : IN STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0);
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of myFunction_linsys_solver_L_i is
    component myFunction_linsys_solver_L_i_ram is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            we0 : IN STD_LOGIC;
            d0 : IN STD_LOGIC_VECTOR;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    myFunction_linsys_solver_L_i_ram_U :  component myFunction_linsys_solver_L_i_ram
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        we0 => we0,
        d0 => d0,
        q0 => q0);

end architecture;


