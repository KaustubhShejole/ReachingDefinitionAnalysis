#include "IR.h"

namespace slim {
// Check if the SSA variable (created using MemorySSA) already exists or not
std::map<std::string, bool> is_ssa_version_available;
} // namespace slim

// Process the llvm instruction and return the corresponding SLIM instruction
BaseInstruction *slim::processLLVMInstruction(llvm::Instruction &instruction) {
  BaseInstruction *base_instruction;

  if (llvm::isa<llvm::AllocaInst>(instruction)) {
    base_instruction = new AllocaInstruction(&instruction);
  } else if (llvm::isa<llvm::LoadInst>(instruction)) {
    base_instruction = new LoadInstruction(&instruction);
  } else if (llvm::isa<llvm::StoreInst>(instruction)) {
    base_instruction = new StoreInstruction(&instruction);
  } else if (llvm::isa<llvm::FenceInst>(instruction)) {
    base_instruction = new FenceInstruction(&instruction);
  } else if (llvm::isa<llvm::AtomicCmpXchgInst>(instruction)) {
    base_instruction = new AtomicCompareChangeInstruction(&instruction);
  } else if (llvm::isa<llvm::AtomicRMWInst>(instruction)) {
    base_instruction = new AtomicModifyMemInstruction(&instruction);
  } else if (llvm::isa<llvm::GetElementPtrInst>(instruction)) {
    base_instruction = new GetElementPtrInstruction(&instruction);
  } else if (llvm::isa<llvm::UnaryOperator>(instruction)) {
    base_instruction = new FPNegationInstruction(&instruction);
  } else if (llvm::isa<llvm::BinaryOperator>(instruction)) {
    base_instruction = new BinaryOperation(&instruction);
  } else if (llvm::isa<llvm::ExtractElementInst>(instruction)) {
    base_instruction = new ExtractElementInstruction(&instruction);
  } else if (llvm::isa<llvm::InsertElementInst>(instruction)) {
    base_instruction = new InsertElementInstruction(&instruction);
  } else if (llvm::isa<llvm::ShuffleVectorInst>(instruction)) {
    base_instruction = new ShuffleVectorInstruction(&instruction);
  } else if (llvm::isa<llvm::ExtractValueInst>(instruction)) {
    base_instruction = new ExtractValueInstruction(&instruction);
  } else if (llvm::isa<llvm::InsertValueInst>(instruction)) {
    base_instruction = new InsertValueInstruction(&instruction);
  } else if (llvm::isa<llvm::TruncInst>(instruction)) {
    base_instruction = new TruncInstruction(&instruction);
  } else if (llvm::isa<llvm::ZExtInst>(instruction)) {
    base_instruction = new ZextInstruction(&instruction);
  } else if (llvm::isa<llvm::SExtInst>(instruction)) {
    base_instruction = new SextInstruction(&instruction);
  } else if (llvm::isa<llvm::FPTruncInst>(instruction)) {
    base_instruction = new TruncInstruction(&instruction);
  } else if (llvm::isa<llvm::FPExtInst>(instruction)) {
    base_instruction = new FPExtInstruction(&instruction);
  } else if (llvm::isa<llvm::FPToUIInst>(instruction)) {
    base_instruction = new FPToIntInstruction(&instruction);
  } else if (llvm::isa<llvm::FPToSIInst>(instruction)) {
    base_instruction = new FPToIntInstruction(&instruction);
  } else if (llvm::isa<llvm::UIToFPInst>(instruction)) {
    base_instruction = new IntToFPInstruction(&instruction);
  } else if (llvm::isa<llvm::SIToFPInst>(instruction)) {
    base_instruction = new IntToFPInstruction(&instruction);
  } else if (llvm::isa<llvm::PtrToIntInst>(instruction)) {
    base_instruction = new PtrToIntInstruction(&instruction);
  } else if (llvm::isa<llvm::IntToPtrInst>(instruction)) {
    base_instruction = new IntToPtrInstruction(&instruction);
  } else if (llvm::isa<llvm::BitCastInst>(instruction)) {
    base_instruction = new BitcastInstruction(&instruction);
  } else if (llvm::isa<llvm::AddrSpaceCastInst>(instruction)) {
    base_instruction = new AddrSpaceInstruction(&instruction);
  } else if (llvm::isa<llvm::ICmpInst>(instruction)) {
    base_instruction = new CompareInstruction(&instruction);
  } else if (llvm::isa<llvm::FCmpInst>(instruction)) {
    base_instruction = new CompareInstruction(&instruction);
  } else if (llvm::isa<llvm::PHINode>(instruction)) {
    base_instruction = new PhiInstruction(&instruction);
  } else if (llvm::isa<llvm::SelectInst>(instruction)) {
    base_instruction = new SelectInstruction(&instruction);
  } else if (llvm::isa<llvm::FreezeInst>(instruction)) {
    base_instruction = new FreezeInstruction(&instruction);
  } else if (llvm::isa<llvm::CallInst>(instruction)) {
    base_instruction = new CallInstruction(&instruction);
  } else if (llvm::isa<llvm::VAArgInst>(instruction)) {
    base_instruction = new VarArgInstruction(&instruction);
  } else if (llvm::isa<llvm::LandingPadInst>(instruction)) {
    base_instruction = new LandingpadInstruction(&instruction);
  } else if (llvm::isa<llvm::CatchPadInst>(instruction)) {
    base_instruction = new CatchpadInstruction(&instruction);
  } else if (llvm::isa<llvm::CleanupPadInst>(instruction)) {
    base_instruction = new CleanuppadInstruction(&instruction);
  } else if (llvm::isa<llvm::ReturnInst>(instruction)) {
    base_instruction = new ReturnInstruction(&instruction);
  } else if (llvm::isa<llvm::BranchInst>(instruction)) {
    base_instruction = new BranchInstruction(&instruction);
  } else if (llvm::isa<llvm::SwitchInst>(instruction)) {
    base_instruction = new SwitchInstruction(&instruction);
  } else if (llvm::isa<llvm::IndirectBrInst>(instruction)) {
    base_instruction = new IndirectBranchInstruction(&instruction);
  } else if (llvm::isa<llvm::InvokeInst>(instruction)) {
    base_instruction = new InvokeInstruction(&instruction);
  } else if (llvm::isa<llvm::CallBrInst>(instruction)) {
    base_instruction = new CallbrInstruction(&instruction);
  } else if (llvm::isa<llvm::ResumeInst>(instruction)) {
    base_instruction = new ResumeInstruction(&instruction);
  } else if (llvm::isa<llvm::CatchSwitchInst>(instruction)) {
    base_instruction = new CatchswitchInstruction(&instruction);
  } else if (llvm::isa<llvm::CatchReturnInst>(instruction)) {
    base_instruction = new CatchreturnInstruction(&instruction);
  } else if (llvm::isa<llvm::CleanupReturnInst>(instruction)) {
    base_instruction = new CleanupReturnInstruction(&instruction);
  } else if (llvm::isa<llvm::UnreachableInst>(instruction)) {
    base_instruction = new UnreachableInstruction(&instruction);
  } else {
    base_instruction = new OtherInstruction(&instruction);
  }

  return base_instruction;
}

// Creates different SSA versions for global and address-taken local variables
// using Memory SSA
void slim::createSSAVersions(std::unique_ptr<llvm::Module> &module) {
  // Fetch the function list of module
  llvm::SymbolTableList<llvm::Function> &function_list =
      module->getFunctionList();

  // Contains the operand object corresponding to every global SSA variable
  std::map<std::string, llvm::Value *> ssa_variable_to_operand;

  // For each function in the module
  for (auto &function : function_list) {
    // Skip the function if it is intrinsic or is not defined in the translation
    // unit
    if (function.isIntrinsic() || function.isDeclaration()) {
      continue;
    }

    llvm::PassBuilder PB;
    llvm::FunctionAnalysisManager *function_analysis_manager =
        new llvm::FunctionAnalysisManager();

    // Register the FunctionAnalysisManager in the pass builder
    PB.registerFunctionAnalyses(*function_analysis_manager);

    llvm::AAManager aa_manager;

    // Add the Basic Alias Analysis provided by LLVM
    aa_manager.registerFunctionAnalysis<llvm::BasicAA>();

    auto alias_analysis = aa_manager.run(function, *function_analysis_manager);

    llvm::DominatorTree *dominator_tree =
        &(*function_analysis_manager)
             .getResult<llvm::DominatorTreeAnalysis>(function);

    llvm::MemorySSA *memory_ssa =
        new llvm::MemorySSA(function, &alias_analysis, dominator_tree);

    // Get the MemorySSAWalker which will be used to query about the clobbering
    // memory definition
    llvm::MemorySSAWalker *memory_ssa_walker = memory_ssa->getWalker();

    std::map<llvm::Value *, bool> is_operand_stack_variable;

    // For each basic block in the function
    for (llvm::BasicBlock &basic_block : function.getBasicBlockList()) {
      std::vector<llvm::Instruction *> instructions;

      for (llvm::Instruction &instruction : basic_block.getInstList()) {
        instructions.push_back(&instruction);
      }

      // For each instruction in the basic block
      for (llvm::Instruction *instruction_ptr : instructions) {
        llvm::Instruction &instruction = *instruction_ptr;
        /*
                                Check if the operand is a address-taken stack
           variable This assumes that the IR has been transformed by mem2reg.
           Since the only variables that are left in "alloca" form (stack
           variables) after mem2reg are the variables whose addresses have been
           taken in some form. The rest of the local variables are promoted to
                                SSA registers by mem2reg.
                        */
        if (llvm::isa<llvm::AllocaInst>(instruction)) {
          is_operand_stack_variable[(llvm::Value *)&instruction] = true;
        }
        // Check if the instruction is a load instruction
        if (llvm::isa<llvm::LoadInst>(instruction)) {
          // Get the clobbering memory access for this load instruction
          llvm::MemoryAccess *clobbering_mem_access =
              memory_ssa_walker->getClobberingMemoryAccess(&instruction);

          std::string ssa_variable_name = "";

          // Check if this the memory access is a MemoryDef or not
          if (llvm::isa<llvm::MemoryDef>(clobbering_mem_access) ||
              llvm::isa<llvm::MemoryPhi>(clobbering_mem_access)) {
            // Cast the MemoryAccess object to MemoryDef object
            llvm::MemoryDef *memory_def =
                llvm::dyn_cast<llvm::MemoryDef, llvm::MemoryAccess *>(
                    clobbering_mem_access);
            llvm::MemoryPhi *memory_phi =
                llvm::dyn_cast<llvm::MemoryPhi, llvm::MemoryAccess *>(
                    clobbering_mem_access);

            unsigned int memory_def_id;

            // Get the memory definition id
            if (llvm::isa<llvm::MemoryDef>(clobbering_mem_access)) {
              memory_def_id = memory_def->getID();
            } else {
              memory_def_id = memory_phi->getID();
            }

            // Fetch the source operand of the load instruction
            llvm::Value *source_operand = instruction.getOperand(0);

            // Check if the source operand is a global variable
            if (llvm::isa<llvm::GlobalVariable>(source_operand) ||
                is_operand_stack_variable[source_operand]) {
              // Based on the memory definition id and global or address-taken
              // local variable name, this is the expected SSA variable name
              ssa_variable_name = function.getName().str() + "_" +
                                  source_operand->getName().str() + "_" +
                                  std::to_string(memory_def_id);

              // Check if the SSA variable (created using MemorySSA) already
              // exists or not
              if (slim::is_ssa_version_available.find(ssa_variable_name) !=
                  slim::is_ssa_version_available.end()) {
                // If the expected SSA variable already exists, then replace the
                // source operand with the corresponding SSA operand
                instruction.setOperand(
                    0, ssa_variable_to_operand[ssa_variable_name]);
              } else {
                // Fetch the basic block iterator
                llvm::BasicBlock::iterator basicblock_iterator =
                    basic_block.begin();

                // Create a new load instruction which loads the value from the
                // memory location to a temporary variable
                llvm::LoadInst *new_load_instr = new llvm::LoadInst(
                    source_operand->getType(), source_operand,
                    "tmp." + ssa_variable_name, &instruction);

                // Create a new alloca instruction for the new SSA version
                llvm::AllocaInst *new_alloca_instr = new llvm::AllocaInst(
                    ((llvm::Value *)new_load_instr)->getType(), 0,
                    ssa_variable_name, new_load_instr);

                // Create a new store instruction to store the value from the
                // new temporary to the new SSA version of global or
                // address-taken local variable
                llvm::StoreInst *new_store_instr = new llvm::StoreInst(
                    (llvm::Value *)new_load_instr,
                    (llvm::Value *)new_alloca_instr, &instruction);

                // Update the map accordingly
                slim::is_ssa_version_available[ssa_variable_name] = true;

                // The value of a instruction corresponds to the result of that
                // instruction
                ssa_variable_to_operand[ssa_variable_name] =
                    (llvm::Value *)new_alloca_instr;

                // Replace the operand of the load instruction with the new SSA
                // version
                instruction.setOperand(
                    0, ssa_variable_to_operand[ssa_variable_name]);
              }
            }
          } else {
            // This is not expected
            llvm_unreachable(
                "Clobbering access is not MemoryDef, which is unexpected!");
          }
        }
      }
    }
  }
}

// Default constructor
slim::IR::IR() {
  this->total_basic_blocks = 0;
  this->total_instructions = 0;
}

// Construct the SLIM IR from module
slim::IR::IR(std::unique_ptr<llvm::Module> &module) {
  this->llvm_module = std::move(module);
  this->total_basic_blocks = 0;
  this->total_instructions = 0;

// Create different SSA versions for globals and address-taken local variables
// if the MemorySSA flag is passed
#ifdef MemorySSAFlag
  slim::createSSAVersions(this->llvm_module);
#endif

  // Fetch the function list of the module
  llvm::SymbolTableList<llvm::Function> &function_list =
      llvm_module->getFunctionList();

  // Keeps track of the temporaries who are already renamed
  std::set<llvm::Value *> renamed_temporaries;

  // For each function in the module
  for (llvm::Function &function : function_list) {
    // Append the pointer to the function to the "functions" list
    if (!function.isIntrinsic() && !function.isDeclaration()) {
      this->functions.push_back(&function);
    } else {
      continue;
    }

#ifdef DiscardPointers
    std::set<llvm::Value *> discarded_result_operands;
#endif

    // For each basic block in the function
    for (llvm::BasicBlock &basic_block : function.getBasicBlockList()) {
      // Create function-basicblock pair
      std::pair<llvm::Function *, llvm::BasicBlock *> func_basic_block{
          &function, &basic_block};

      this->basic_block_to_id[&basic_block] = this->total_basic_blocks;

      this->total_basic_blocks++;

      // For each instruction in the basic block
      for (llvm::Instruction &instruction : basic_block.getInstList()) {
        if (instruction.isDebugOrPseudoInst()) {
          continue;
        }

        // Ensure that all temporaries have unique name (globally) by appending
        // the function name after the temporary name
        for (unsigned i = 0; i < instruction.getNumOperands(); i++) {
          llvm::Value *operand_i = instruction.getOperand(i);

          if (llvm::isa<llvm::GlobalValue>(operand_i))
            continue;

          if (operand_i->hasName() && renamed_temporaries.find(operand_i) ==
                                          renamed_temporaries.end()) {
            llvm::StringRef old_name = operand_i->getName();
            operand_i->setName(old_name + "_" + function.getName());
            renamed_temporaries.insert(operand_i);
          }
        }

        BaseInstruction *base_instruction =
            slim::processLLVMInstruction(instruction);

#ifdef DiscardPointers
        bool is_discarded = false;

        for (unsigned i = 0; i < base_instruction->getNumOperands(); i++) {
          SLIMOperand *operand_i = base_instruction->getOperand(i).first;

          if (!operand_i || !operand_i->getValue())
            continue;
          if (operand_i->isPointerInLLVM() ||
              (operand_i->getValue() &&
               discarded_result_operands.find(operand_i->getValue()) !=
                   discarded_result_operands.end())) {
            is_discarded = true;
            break;
          }
        }

        if (is_discarded && base_instruction->getResultOperand().first &&
            base_instruction->getResultOperand().first->getValue()) {
          discarded_result_operands.insert(
              base_instruction->getResultOperand().first->getValue());
          continue;
        } else if (is_discarded) {
          // Ignore the instruction (because it is using the discarded value)
          continue;
        }
#endif

        if (base_instruction->getInstructionType() == InstructionType::CALL) {
          CallInstruction *call_instruction =
              (CallInstruction *)base_instruction;

          if (!call_instruction->isIndirectCall() &&
              !call_instruction->getCalleeFunction()->isDeclaration()) {
            for (unsigned arg_i = 0;
                 arg_i < call_instruction->getNumFormalArguments(); arg_i++) {
              llvm::Argument *formal_argument =
                  call_instruction->getFormalArgument(arg_i);
              SLIMOperand *formal_slim_argument =
                  OperandRepository::getSLIMOperand(formal_argument);

              if (!formal_slim_argument) {
                formal_slim_argument = new SLIMOperand(formal_argument);
                OperandRepository::setSLIMOperand(formal_argument,
                                                  formal_slim_argument);

                if (formal_argument->hasName() &&
                    renamed_temporaries.find(formal_argument) ==
                        renamed_temporaries.end()) {
                  llvm::StringRef old_name = formal_argument->getName();
                  formal_argument->setName(
                      old_name + "_" +
                      call_instruction->getCalleeFunction()->getName());
                  renamed_temporaries.insert(formal_argument);
                }

                formal_slim_argument->setFormalArgument();
              }

              LoadInstruction *new_load_instr = new LoadInstruction(
                  &llvm::cast<llvm::CallInst>(instruction),
                  formal_slim_argument,
                  call_instruction->getOperand(arg_i).first);

              // The initial value of total instructions is 0 and it is
              // incremented after every instruction
              long long instruction_id = slim::IR::total_instructions;

              // Increment the total instructions count
              this->total_instructions++;

              new_load_instr->setInstructionId(instruction_id);

              this->func_bb_to_inst_id[func_basic_block].push_back(
                  instruction_id);

              // Map the instruction id to the corresponding SLIM instruction
              this->inst_id_to_object[instruction_id] = new_load_instr;
            }
          }
        }

        // The initial value of total instructions is 0 and it is incremented
        // after every instruction
        long long instruction_id = this->total_instructions;

        // Increment the total instructions count
        this->total_instructions++;

        base_instruction->setInstructionId(instruction_id);

        this->func_bb_to_inst_id[func_basic_block].push_back(instruction_id);

        // Map the instruction id to the corresponding SLIM instruction
        this->inst_id_to_object[instruction_id] = base_instruction;

        // Check if the instruction is a "Return" instruction
        if (base_instruction->getInstructionType() == InstructionType::RETURN) {
          // As we are using the 'mergereturn' pass, there is only one return
          // statement in every function and therefore, we will have only 1
          // return operand which we store in the function_return_operand map
          ReturnInstruction *return_instruction =
              (ReturnInstruction *)base_instruction;

          if (return_instruction->getNumOperands() == 0) {
            OperandRepository::setFunctionReturnOperand(&function, nullptr);
          } else {
            OperandRepository::setFunctionReturnOperand(
                &function, return_instruction->getReturnOperand());
          }
        }
      }
    }
  }

  llvm::outs() << "Total number of functions: " << functions.size() << "\n";
  llvm::outs() << "Total number of basic blocks: " << total_basic_blocks
               << "\n";
  llvm::outs() << "Total number of instructions: " << total_instructions
               << "\n";
}

// Returns the total number of instructions across all the functions and basic
// blocks
long long slim::IR::getTotalInstructions() { return this->total_instructions; }

// Return the total number of functions in the module
unsigned slim::IR::getNumberOfFunctions() { return this->functions.size(); }

// Return the total number of basic blocks in the module
long long slim::IR::getNumberOfBasicBlocks() {
  return this->total_basic_blocks;
}

// Returns the pointer to llvm::Function for the function at the given index
llvm::Function *slim::IR::getLLVMFunction(unsigned index) {
  // Make sure that the index is not out-of-bounds
  assert(index >= 0 && index < this->getNumberOfFunctions());

  return this->functions[index];
}

// Add instructions for function-basicblock pair (used by the LegacyIR)
void slim::IR::addFuncBasicBlockInstructions(llvm::Function *function,
                                             llvm::BasicBlock *basic_block) {
  // Create function-basicblock pair
  std::pair<llvm::Function *, llvm::BasicBlock *> func_basic_block{function,
                                                                   basic_block};

  // For each instruction in the basic block
  for (llvm::Instruction &instruction : basic_block->getInstList()) {
    BaseInstruction *base_instruction =
        slim::processLLVMInstruction(instruction);

    // Get the instruction id
    long long instruction_id = this->total_instructions;

    // Increment the total instructions count
    this->total_instructions++;

    base_instruction->setInstructionId(instruction_id);

    this->func_bb_to_inst_id[func_basic_block].push_back(instruction_id);

    // Map the instruction id to the corresponding SLIM instruction
    this->inst_id_to_object[instruction_id] = base_instruction;
  }
}

// Return the function-basicblock to instructions map (required by the LegacyIR)
std::map<std::pair<llvm::Function *, llvm::BasicBlock *>,
         std::list<long long>> &
slim::IR::getFuncBBToInstructions() {
  return this->func_bb_to_inst_id;
}

// Get the instruction id to SLIM instruction map (required by the LegacyIR)
std::map<long long, BaseInstruction *> &slim::IR::getIdToInstructionsMap() {
  return this->inst_id_to_object;
}

// Returns the first instruction id in the instruction list of the given
// function-basicblock pair
long long slim::IR::getFirstIns(llvm::Function *function,
                                llvm::BasicBlock *basic_block) {
  // Make sure that the list corresponding to the function-basicblock pair
  // exists
  assert(this->func_bb_to_inst_id.find({function, basic_block}) !=
         this->func_bb_to_inst_id.end());

  auto result = func_bb_to_inst_id.find({function, basic_block});

#ifndef DISABLE_IGNORE_EFFECT
  auto it = result->second.begin();

  while (it != result->second.end() &&
         this->inst_id_to_object[*it]->isIgnored()) {
    it++;
  }

  return (it == result->second.end() ? -1 : (*it));
#else
  return result->second.front();
#endif
}

// Returns the last instruction id in the instruction list of the given
// function-basicblock pair
long long slim::IR::getLastIns(llvm::Function *function,
                               llvm::BasicBlock *basic_block) {
  // Make sure that the list corresponding to the function-basicblock pair
  // exists
  assert(this->func_bb_to_inst_id.find({function, basic_block}) !=
         this->func_bb_to_inst_id.end());

  auto result = func_bb_to_inst_id.find({function, basic_block});

#ifndef DISABLE_IGNORE_EFFECT
  auto it = result->second.rbegin();

  while (it != result->second.rend() &&
         this->inst_id_to_object[*it]->isIgnored()) {
    it++;
  }

  return (it == result->second.rend() ? -1 : (*it));
#else
  return result->second.back();
#endif
}

// Returns the reversed instruction list for a given function and a basic block
std::list<long long>
slim::IR::getReverseInstList(llvm::Function *function,
                             llvm::BasicBlock *basic_block) {
  // Make sure that the list corresponding to the function-basicblock pair
  // exists
  assert(this->func_bb_to_inst_id.find({function, basic_block}) !=
         this->func_bb_to_inst_id.end());

  std::list<long long> inst_list =
      this->func_bb_to_inst_id[{function, basic_block}];

  inst_list.reverse();

  return inst_list;
}

// Returns the reversed instruction list (for the list passed as an argument)
std::list<long long>
slim::IR::getReverseInstList(std::list<long long> inst_list) {
  inst_list.reverse();
  return inst_list;
}

// Get SLIM instruction from the instruction index
BaseInstruction *slim::IR::getInstrFromIndex(long long index) {
  return this->inst_id_to_object[index];
}

// Get basic block id
long long slim::IR::getBasicBlockId(llvm::BasicBlock *basic_block) {
  assert(this->basic_block_to_id.find(basic_block) !=
         this->basic_block_to_id.end());

  return this->basic_block_to_id[basic_block];
}

// Inserts instruction at the front of the basic block (only in this
// abstraction)
void slim::IR::insertInstrAtFront(BaseInstruction *instruction,
                                  llvm::BasicBlock *basic_block) {
  assert(instruction != nullptr && basic_block != nullptr);

  instruction->setInstructionId(this->total_instructions);

  this->func_bb_to_inst_id[std::make_pair(basic_block->getParent(),
                                          basic_block)]
      .push_front(this->total_instructions);

  this->inst_id_to_object[this->total_instructions] = instruction;

  this->total_instructions++;
}

// Inserts instruction at the end of the basic block (only in this abstraction)
void slim::IR::insertInstrAtBack(BaseInstruction *instruction,
                                 llvm::BasicBlock *basic_block) {
  assert(instruction != nullptr && basic_block != nullptr);

  instruction->setInstructionId(this->total_instructions);

  this->func_bb_to_inst_id[std::make_pair(basic_block->getParent(),
                                          basic_block)]
      .push_back(this->total_instructions);

  this->inst_id_to_object[this->total_instructions] = instruction;

  this->total_instructions++;
}

// Optimize the IR (please use only when you are using the MemorySSAFlag)
slim::IR *slim::IR::optimizeIR() {
  // Create the new slim::IR object which would contain the IR instructions
  // after optimization
  slim::IR *optimized_slim_ir = new slim::IR();

  // errs() << "funcBBInsMap size: " << funcBBInsMap.size() << "\n";

  // Now, we are ready to do the load-store optimization
  for (auto func_basicblock_instr_entry : this->func_bb_to_inst_id) {
    // Add the function-basic-block entry in optimized_slim_ir
    optimized_slim_ir->func_bb_to_inst_id[func_basicblock_instr_entry.first] =
        std::list<long long>{};

    std::list<long long> &instruction_list = func_basicblock_instr_entry.second;

    // errs() << "instruction_list size: " << instruction_list.size() << "\n";

    long long temp_instr_counter = 0;
    std::map<llvm::Value *, BaseInstruction *> temp_token_to_instruction;
    std::map<long long, BaseInstruction *> temp_instructions;
    std::map<BaseInstruction *, long long> temp_instruction_ids;

    for (auto instruction_id : instruction_list) {
      // errs() << "Instruction id : " << instruction_id << "\n";
      BaseInstruction *slim_instruction =
          this->inst_id_to_object[instruction_id];

      // Check if the corresponding LLVM instruction is a Store Instruction
      if (slim_instruction->getInstructionType() == InstructionType::STORE &&
          slim_instruction->getNumOperands() == 1 &&
          slim_instruction->getResultOperand().first != nullptr) {
        // Retrieve the RHS operand of the SLIM instruction (corresponding to
        // the LLVM IR store instruction)
        std::pair<SLIMOperand *, int> slim_instr_rhs =
            slim_instruction->getOperand(0);

        // Extract the value and its indirection level from slim_instr_rhs
        llvm::Value *slim_instr_rhs_value = slim_instr_rhs.first->getValue();
        int token_indirection = slim_instr_rhs.second;

        // Check if the RHS Value is defined in an earlier SLIM statement
        if (temp_token_to_instruction.find(slim_instr_rhs_value) !=
            temp_token_to_instruction.end()) {
          // Get the instruction (in the unoptimized SLIM IR)
          BaseInstruction *value_def_instr =
              temp_token_to_instruction[slim_instr_rhs_value];
          long long value_def_index = temp_instruction_ids[value_def_instr];

          // Check if the statement is a load instruction
          bool is_load_instr = (llvm::isa<llvm::LoadInst>(
              value_def_instr->getLLVMInstruction()));

          // Get the indirection level of the LHS operand in the load
          // instruction
          int map_entry_indirection =
              value_def_instr->getResultOperand().second;

          // Get the indirection level of the RHS operand in the load
          // instruction
          int map_entry_rhs_indirection = value_def_instr->getOperand(0).second;

          // Adjust the indirection level
          int distance = token_indirection - map_entry_indirection +
                         map_entry_rhs_indirection;

          // Check if the RHS is a SSA variable (created using MemorySSA)
          bool is_rhs_global_ssa_variable =
              (slim::is_ssa_version_available.find(
                   slim_instr_rhs_value->getName().str()) !=
               slim::is_ssa_version_available.end());

          // Modify the RHS operand with the new indirection level if it does
          // not exceed 2
          if (is_load_instr && (distance >= 0 && distance <= 2) &&
              !is_rhs_global_ssa_variable) {
            // errs() << slim_instr_rhs.first->getName() << " = name\n";

            // Set the indirection level of the RHS operand to the adjusted
            // indirection level
            value_def_instr->setRHSIndirection(0, distance);

            // Update the RHS operand of the store instruction
            slim_instruction->setOperand(0, value_def_instr->getOperand(0));

            // Remove the existing entries
            temp_token_to_instruction.erase(slim_instr_rhs_value);
            temp_instructions.erase(value_def_index);
          }
        } else {
          // errs() << "RHS is not present as LHS!\n";
        }

        // Check whether the LHS operand can be replaced
        std::pair<SLIMOperand *, int> slim_instr_lhs =
            slim_instruction->getResultOperand();

        // Extract the value and its indirection level from slim_instr_rhs
        llvm::Value *slim_instr_lhs_value = slim_instr_lhs.first->getValue();
        token_indirection = slim_instr_lhs.second;

        // Check if the LHS Value is defined in an earlier SLIM statement
        if (temp_token_to_instruction.find(slim_instr_lhs_value) !=
            temp_token_to_instruction.end()) {
          // Get the instruction (in the unoptimized SLIM IR)
          BaseInstruction *value_def_instr =
              temp_token_to_instruction[slim_instr_lhs_value];
          long long value_def_index = temp_instruction_ids[value_def_instr];

          // Check if the statement is a load instruction
          bool is_load_instr = (llvm::isa<llvm::LoadInst>(
              value_def_instr->getLLVMInstruction()));

          // Get the indirection level of the LHS operand in the load
          // instruction
          int map_entry_indirection =
              value_def_instr->getResultOperand().second;

          // Get the indirection level of the RHS operand in the load
          // instruction
          int map_entry_rhs_indirection = value_def_instr->getOperand(0).second;

          // Adjust the indirection level
          int distance = token_indirection - map_entry_indirection +
                         map_entry_rhs_indirection;

          // Check if the RHS is a SSA variable (created using MemorySSA)
          bool is_rhs_global_ssa_variable =
              (slim::is_ssa_version_available.find(
                   slim_instr_lhs_value->getName().str()) !=
               slim::is_ssa_version_available.end());

          // Modify the RHS operand with the new indirection level if it does
          // not exceed 2
          if (is_load_instr && (distance >= 0 && distance <= 2) &&
              !is_rhs_global_ssa_variable) {
            // errs() << slim_instr_rhs.first->getName() << " = name\n";

            // Set the indirection level of the RHS operand to the adjusted
            // indirection level
            value_def_instr->setRHSIndirection(0, distance);

            // Update the result operand of the store instruction
            slim_instruction->setResultOperand(value_def_instr->getOperand(0));

            // Remove the existing entries
            temp_token_to_instruction.erase(slim_instr_lhs_value);
            temp_instructions.erase(value_def_index);
          }
        }

        // Add the SLIM instruction (whether modified or not)
        if (slim_instruction->getInstructionType() == InstructionType::LOAD)
          temp_token_to_instruction[slim_instruction->getResultOperand()
                                        .first->getValue()] = slim_instruction;
        temp_instructions[temp_instr_counter] = slim_instruction;
        temp_instruction_ids[slim_instruction] = temp_instr_counter;
        temp_instr_counter++;
      } else {
        // errs() << "Size != 1\n";
        //  Add the SLIM instruction
        // Add the SLIM instruction (whether modified or not)
        if (slim_instruction->getInstructionType() == InstructionType::LOAD)
          temp_token_to_instruction[slim_instruction->getResultOperand()
                                        .first->getValue()] = slim_instruction;
        temp_instructions[temp_instr_counter] = slim_instruction;
        temp_instruction_ids[slim_instruction] = temp_instr_counter;
        temp_instr_counter++;
      }
    }

    // Now, we have the final list of optimized instructions in this basic
    // block. So, we insert the instructions in the optimized global
    // instructions list and the instruction indices (after the optimization) in
    // the func_basic_block_optimized_instrs map
    for (auto temp_instruction : temp_instructions) {
      temp_instruction.second->setInstructionId(
          optimized_slim_ir->total_instructions);
      optimized_slim_ir->func_bb_to_inst_id[func_basicblock_instr_entry.first]
          .push_back(optimized_slim_ir->total_instructions);
      optimized_slim_ir
          ->inst_id_to_object[optimized_slim_ir->total_instructions] =
          temp_instruction.second;
      optimized_slim_ir->total_instructions++;
    }

    optimized_slim_ir
        ->basic_block_to_id[func_basicblock_instr_entry.first.second] =
        optimized_slim_ir->total_basic_blocks++;
  }

  return optimized_slim_ir;
}
/*
// Dump the IR with detailed instruction information
void slim::IR::dumpIR()
{
    // Keeps track whether the function details have been printed or not
    std::unordered_map<llvm::Function *, bool> func_visited;

    // Iterate over the function basic block map
    for (auto &entry : this->func_bb_to_inst_id)
    {
        llvm::Function *func = entry.first.first;
        llvm::BasicBlock *basic_block = entry.first.second;

        // Check if the function details are already printed and if not, print
the details and mark
        // the function as visited
        if (func_visited.find(func) == func_visited.end())
        {
            if (func->getSubprogram())
                llvm::outs() << "[" << func->getSubprogram()->getFilename() <<
"] ";

            llvm::outs() << "Function: " << func->getName() << "\n";
            llvm::outs() << "A-------------------------------------A" << "\n";

            // Mark the function as visited
            func_visited[func] = true;
        }

        // Print the basic block name
        llvm::outs() << "Basic block " << this->getBasicBlockId(basic_block) <<
": " << basic_block->getName() << " (Predecessors::: "; llvm::outs() << "[";

        // Print the names of predecessor basic blocks
        for (auto pred = llvm::pred_begin(basic_block); pred !=
llvm::pred_end(basic_block); pred++)
        {
            llvm::outs() << (*pred)->getName();

            if (std::next(pred) != ((llvm::pred_end(basic_block))))
            {
                llvm::outs() << ",,, ";
            }
        }

        llvm::outs() << "])\n";

        for (long long instruction_id : entry.second)
        {
            BaseInstruction *instruction = inst_id_to_object[instruction_id];

            // Print instruction type
            llvm::outs() << "Instruction Type: " <<
instruction->getInstructionType() << "\n";

            // Print operands
            llvm::outs() << "Operands:\n";
            for (const auto &operandPair : instruction->getOperands())
            {
                llvm::outs() << "Operand: ";
                operandPair.first->printOperand(llvm::outs());
                llvm::outs() << " (Indirection Level: " << operandPair.second <<
")\n";
            }

            // Print operator (if applicable)
            if (instruction->getInstructionType() == "BinaryOperation")
            {
                BinaryOperation *binaryOp =
llvm::cast<BinaryOperation>(instruction); llvm::outs() << "Operator: " <<
binaryOp->getOperationType() << "\n";
            }

            // Print the instruction itself
            instruction->printInstruction();
        }

        llvm::outs() << "\n\n";
    }
}

// Dump the IR
void slim::IR::dumpIR() {
    // Keeps track of whether the function details have been printed or not
    std::unordered_map<llvm::Function *, bool> func_visited;

    // Iterate over the function basic block map
    for (auto &entry : this->func_bb_to_inst_id) {
        llvm::Function *func = entry.first.first;
        llvm::BasicBlock *basic_block = entry.first.second;

        // Check if the function details are already printed and if not, print
the details and mark
        // the function as visited
        if (func_visited.find(func) == func_visited.end()) {
            if (func->getSubprogram())
                llvm::outs() << "[" << func->getSubprogram()->getFilename() <<
"] ";

            llvm::outs() << "Function: " << func->getName() << "\n";
            llvm::outs() << "A-------------------------------------A" << "\n";

            // Mark the function as visited
            func_visited[func] = true;
        }

        // Print the basic block name
        llvm::outs() << "Basic block " << this->getBasicBlockId(basic_block) <<
": " << basic_block->getName() << " (Predecessors::: "; llvm::outs() << "[";

        // Print the names of predecessor basic blocks
        for (auto pred = llvm::pred_begin(basic_block); pred !=
llvm::pred_end(basic_block); pred++) { llvm::outs() << (*pred)->getName();

            if (std::next(pred) != ((llvm::pred_end(basic_block)))) {
                llvm::outs() << ",,, ";
            }
        }

        llvm::outs() << "])\n";

        for (long long instruction_id : entry.second)
        {
            BaseInstruction *instruction = inst_id_to_object[instruction_id];

            // Print the instruction type
            llvm::outs() << "Instruction Type: " <<
instruction->getInstructionTypeString() << "\n";


            // Print operands and operators
            llvm::outs() << "Operands: ";
            for (const auto &operandPair : instruction->getOperands()) {
                SLIMOperand *operand = operandPair.first;
                llvm::outs() << operand->toString() << ", ";
            }
            llvm::outs() << "\n";

            // Print the operator for binary operations (if applicable)
            if (instruction->getInstructionType() ==
InstructionType::BINARY_OPERATION) { BinaryOperation *binaryOp =
dynamic_cast<BinaryOperation *>(instruction); if (binaryOp) { llvm::outs() <<
"Operator: "; switch (binaryOp->getOperationType()) { case
SLIMBinaryOperator::ADD: llvm::outs() << "+"; break; case
SLIMBinaryOperator::SUB: llvm::outs() << "-"; break; case
SLIMBinaryOperator::MUL: llvm::outs() << "*"; break;
                        // Add cases for other operators as needed
                        default:
                            llvm::outs() << "Unknown";
                    }
                    llvm::outs() << "\n";
                }
            }

            llvm::outs() << "\n";
        }

        llvm::outs() << "\n\n";
    }
}*/
/**/

// Function to print LLVM IR details for a function
void slim::IR::printFunctionDetails(
    llvm::Function *func,
    std::unordered_map<llvm::Function *, bool> &func_visited) {
  if (func_visited.find(func) == func_visited.end()) {
    if (func->getSubprogram())
      llvm::outs() << "[" << func->getSubprogram()->getFilename() << "] ";

    llvm::outs() << "Function: " << func->getName() << "\n";
    llvm::outs() << "-------------------------------------"
                 << "\n";
    func_visited[func] = true;
  }
}
// Function to print basic block details
void slim::IR::printBasicBlockDetails(
    llvm::BasicBlock *basic_block,
    std::unordered_map<llvm::Function *, bool> &func_visited) {
  llvm::outs() << "Basic block " << this->getBasicBlockId(basic_block) << ": "
               << basic_block->getName() << " (Predecessors: [";

  for (auto pred = llvm::pred_begin(basic_block);
       pred != llvm::pred_end(basic_block); pred++) {
    llvm::outs() << (*pred)->getName();

    if (std::next(pred) != llvm::pred_end(basic_block)) {
      llvm::outs() << ",";
    }
  }

  llvm::outs() << "])\n";
}

// Define the global variable for definitionSets
std::unordered_map<std::string, std::vector<int>> definitionSets;
struct GenSet {
  std::unordered_map<std::string, int> definitions;
};
struct KillSet {
  std::unordered_map<std::string, std::vector<int>> definitions;
};
struct OurSet {
  std::unordered_map<std::string, std::vector<int>> definitions;
};
std::unordered_map<int, std::vector<int>> blockPredecessors;

// Define the Definition type, you can modify it as needed
struct Definition {
  std::string variableName;
  int index;

  bool operator==(const Definition &other) const {
    return variableName == other.variableName && index == other.index;
  }
};

namespace std {
template <> struct hash<Definition> {
  size_t operator()(const Definition &definition) const {
    return hash<string>()(definition.variableName) ^
           hash<int>()(definition.index);
  }
};
} // namespace std

class BasicBlocksInOutInformation {
public:
  // Constructor that initializes InSets and OutSets for numBasicBlocks
  BasicBlocksInOutInformation() {}
  BasicBlocksInOutInformation(int numBasicBlocks)
      : InSets(numBasicBlocks), OutSets(numBasicBlocks) {}

  // Function to add a definition to InSet at a specific index
  void addToInSet(int blockIndex, const Definition &definition) {
    InSets[blockIndex].insert(definition);
  }

  // Function to add a definition to OutSet at a specific index
  void addToOutSet(int blockIndex, const Definition &definition) {
    OutSets[blockIndex].insert(definition);
  }

  // Function to print InSet at a specific index
  void printInSet(int blockIndex) {
    std::cout << "InSet[" << blockIndex << "]: ";
    for (const Definition &definition : InSets[blockIndex]) {
      std::cout << definition.variableName << "_" << definition.index << " ,";
    }
    std::cout << "\n";
  }

  // Function to print OutSet at a specific index
  void printOutSet(int blockIndex) {
    std::cout << "OutSet[" << blockIndex << "]: ";
    for (const Definition &definition : OutSets[blockIndex]) {
      std::cout << definition.variableName << "_" << definition.index << ",";
    }
    std::cout << "\n";
  }
  BasicBlocksInOutInformation &
  operator=(const BasicBlocksInOutInformation &rhs) {
    if (this == &rhs) {
      // Self-assignment, no need to do anything
      return *this;
    }

    // Copy InSets and OutSets from rhs to this
    this->InSets = rhs.InSets;
    this->OutSets = rhs.OutSets;

    // Return a reference to this object
    return *this;
  }
  bool operator==(const BasicBlocksInOutInformation &other) const {
    // Check if InSets and OutSets of both objects are equal
    return (InSets == other.InSets) && (OutSets == other.OutSets);
  }
  bool operator!=(const BasicBlocksInOutInformation &other) const {
    // Check if InSets and OutSets of both objects are equal
    return (InSets != other.InSets) && (OutSets != other.OutSets);
  }
  // private:
  // Arrays of InSet and OutSet for each basic block
  std::vector<std::unordered_set<Definition>> InSets;
  std::vector<std::unordered_set<Definition>> OutSets;
};

// Function to print the definition sets
void printKillSet(const KillSet &killSet) {
  std::cout << "\nKill Set: ";
  for (const auto &variable : killSet.definitions) {
    const std::string &variableName = variable.first;
    const std::vector<int> &definitions = variable.second;

    for (int definition : definitions) {
      std::cout << variableName << "_" << definition << ",";
    }
    std::cout << "\n";
  }
}
// Function to print the GenSet
void printGenSet(const GenSet &genSet) {
  std::cout << "\nGen Set: ";
  for (const auto &variable : genSet.definitions) {
    const std::string &variableName = variable.first;
    int definition = variable.second;

    std::cout << variableName << "_" << definition << ", ";
  }
  std::cout << "\n";
}
// Function to print the OurSet
void printOurSet(const OurSet &ourSet, const char *str) {
  std::cout << str << " \n";
  for (const auto &variable : ourSet.definitions) {
    const std::string &variableName = variable.first;
    const std::vector<int> &definitions = variable.second;

    for (int definition : definitions) {
      std::cout << variableName << "_" << definition << ",";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}
// Function to add a new definition for a variable or create a definition set if
// not present
void addDefinition(const std::string &variableName) {
  // Check if the variable already exists in the map
  if (definitionSets.find(variableName) == definitionSets.end()) {
    // Create a new definition set if the variable doesn't exist
    definitionSets[variableName] = std::vector<int>();
  }

  // Automatically generate a new value as the next index
  int newIndex = definitionSets[variableName].size();

  definitionSets[variableName].push_back(newIndex);
}
// Function to add a new definition for a variable or create a definition set if
// not present
void addinitialDefinition(const std::string &variableName) {
  // Check if the variable already exists in the map
  if (definitionSets.find(variableName) == definitionSets.end()) {
    // Create a new definition set if the variable doesn't exist
    definitionSets[variableName] = std::vector<int>();
  }

  // Automatically generate a new value as the next index
  int newIndex = definitionSets[variableName].size();
  if (newIndex == 0) {
    definitionSets[variableName].push_back(newIndex);
  }
}
// Function to print the definition sets
void printDefinitionSets() {
  for (const auto &variable : definitionSets) {
    const std::string &variableName = variable.first;
    const std::vector<int> &definitions = variable.second;

    std::cout << "Definition set for " << variableName << ": ";
    for (int definition : definitions) {
      std::cout << definition << " ";
    }
    std::cout << "\n";
  }
}

// Function to print instruction details
void slim::IR::printInstructionDetails(BaseInstruction *instruction) {
  llvm::outs() << "----------------------------------------------------------"
               << "\n";

  // Print the instruction type
  // llvm::outs() << "Instruction Type: ";
  instruction->printInstruction();
  switch (instruction->getInstructionType()) {
  case InstructionType::ALLOCA:
    llvm::outs() << "ALLOCA\n";

    break;
  case InstructionType::LOAD:
    llvm::outs() << "LOAD\n";
    break;
  case InstructionType::STORE:
    llvm::outs() << "STORE\n";
    break;
  case InstructionType::FENCE:
    llvm::outs() << "FENCE\n";
    break;
  case InstructionType::ATOMIC_COMPARE_CHANGE:
    llvm::outs() << "ATOMIC_COMPARE_CHANGE\n";
    break;
  case InstructionType::ATOMIC_MODIFY_MEM:
    llvm::outs() << "ATOMIC_MODIFY_MEM\n";
    break;
  case InstructionType::GET_ELEMENT_PTR:
    llvm::outs() << "GET_ELEMENT_PTR\n";
    break;
  case InstructionType::FP_NEGATION:
    llvm::outs() << "FP_NEGATION\n";
    break;
  case InstructionType::BINARY_OPERATION: {
    llvm::outs() << "BINARY_OPERATION\n";
    // Handle binary operation here
    // if (auto binaryOp = dynamic_cast<BinaryOperation *>(instruction)) {
    //     // Call the printInstruction method for binary operation
    //     binaryOp->printInstruction();
    // }
  } break;
  case InstructionType::EXTRACT_ELEMENT:
    llvm::outs() << "EXTRACT_ELEMENT\n";
    break;
  case InstructionType::INSERT_ELEMENT:
    llvm::outs() << "INSERT_ELEMENT\n";
    break;
  case InstructionType::SHUFFLE_VECTOR:
    llvm::outs() << "SHUFFLE_VECTOR\n";
    break;
  case InstructionType::EXTRACT_VALUE:
    llvm::outs() << "EXTRACT_VALUE\n";
    break;
  case InstructionType::INSERT_VALUE:
    llvm::outs() << "INSERT_VALUE\n";
    break;
  case InstructionType::TRUNC:
    llvm::outs() << "TRUNC\n";
    break;
  case InstructionType::ZEXT:
    llvm::outs() << "ZEXT\n";
    break;
  case InstructionType::SEXT:
    llvm::outs() << "SEXT\n";
    break;
  case InstructionType::FPEXT:
    llvm::outs() << "FPEXT\n";
    break;
  case InstructionType::FP_TO_INT:
    llvm::outs() << "FP_TO_INT\n";
    break;
  case InstructionType::INT_TO_FP:
    llvm::outs() << "INT_TO_FP\n";
    break;
  case InstructionType::PTR_TO_INT:
    llvm::outs() << "PTR_TO_INT\n";
    break;
  case InstructionType::INT_TO_PTR:
    llvm::outs() << "INT_TO_PTR\n";
    break;
  case InstructionType::BITCAST:
    llvm::outs() << "BITCAST\n";
    break;
  case InstructionType::ADDR_SPACE:
    llvm::outs() << "ADDR_SPACE\n";
    break;
  case InstructionType::COMPARE:
    llvm::outs() << "COMPARE\n";
    break;
  case InstructionType::PHI:
    llvm::outs() << "PHI\n";
    break;
  case InstructionType::SELECT:
    llvm::outs() << "SELECT\n";
    break;
  case InstructionType::FREEZE:
    llvm::outs() << "FREEZE\n";
    break;
  case InstructionType::CALL:
    llvm::outs() << "CALL\n";
    break;
  case InstructionType::VAR_ARG:
    llvm::outs() << "VAR_ARG\n";
    break;
  case InstructionType::LANDING_PAD:
    llvm::outs() << "LANDING_PAD\n";
    break;
  case InstructionType::CATCH_PAD:
    llvm::outs() << "CATCH_PAD\n";
    break;
  case InstructionType::CLEANUP_PAD:
    llvm::outs() << "CLEANUP_PAD\n";
    break;
  case InstructionType::RETURN:
    llvm::outs() << "RETURN\n";
    break;
  case InstructionType::BRANCH:
    llvm::outs() << "BRANCH\n";
    break;
  case InstructionType::SWITCH:
    llvm::outs() << "SWITCH\n";
    break;
  case InstructionType::INDIRECT_BRANCH:
    llvm::outs() << "INDIRECT_BRANCH\n";
    break;
  case InstructionType::INVOKE:
    llvm::outs() << "INVOKE\n";
    break;
  case InstructionType::CALL_BR:
    llvm::outs() << "CALL_BR\n";
    break;
  case InstructionType::RESUME:
    llvm::outs() << "RESUME\n";
    break;
  case InstructionType::CATCH_SWITCH:
    llvm::outs() << "CATCH_SWITCH\n";
    break;
  case InstructionType::CATCH_RETURN:
    llvm::outs() << "CATCH_RETURN\n";
    break;
  case InstructionType::CLEANUP_RETURN:
    llvm::outs() << "CLEANUP_RETURN\n";
    break;
  case InstructionType::UNREACHABLE:
    llvm::outs() << "UNREACHABLE\n";
    break;
  case InstructionType::OTHER:
    llvm::outs() << "OTHER\n";
    break;
  case InstructionType::NOT_ASSIGNED:
    llvm::outs() << "NOT_ASSIGNED\n";
    break;
  }
  // instruction->printInstruction();
  if (instruction->getInstructionType() < 33) {
    // Determine the instruction type and print it
    OperandType operandType =
        instruction->getResultOperand().first->getOperandType();
    switch (operandType) {
    case OperandType::GEP_OPERATOR:
      llvm::outs() << "GEP Operator";
      break;
    case OperandType::ADDR_SPACE_CAST_OPERATOR:
      llvm::outs() << "AddrSpaceCast Operator";
      break;
    case OperandType::BITCAST_OPERATOR:
      llvm::outs() << "Bitcast Operator";
      break;
    case OperandType::PTR_TO_INT_OPERATOR:
      llvm::outs() << "PtrToInt Operator";
      break;
    case OperandType::ZEXT_OPERATOR:
      llvm::outs() << "ZExt Operator";
      break;
    case OperandType::FP_MATH_OPERATOR:
      llvm::outs() << "FP Math Operator";
      break;
    case OperandType::BLOCK_ADDRESS:
      llvm::outs() << "Block Address";
      break;
    case OperandType::CONSTANT_AGGREGATE:
      llvm::outs() << "Constant Aggregate";
      break;
    case OperandType::CONSTANT_DATA_SEQUENTIAL:
      llvm::outs() << "Constant Data Sequential";
      break;
    case OperandType::CONSTANT_POINTER_NULL:
      llvm::outs() << "Constant Pointer Null";
      break;
    case OperandType::CONSTANT_TOKEN_NONE:
      llvm::outs() << "Constant Token None";
      break;
    case OperandType::UNDEF_VALUE:
      llvm::outs() << "Undef Value";
      break;
    case OperandType::CONSTANT_INT:
      llvm::outs() << "Constant Int";
      break;
    case OperandType::CONSTANT_FP:
      llvm::outs() << "Constant FP";
      break;
    case OperandType::DSO_LOCAL_EQUIVALENT:
      llvm::outs() << "DSO Local Equivalent";
      break;
    case OperandType::GLOBAL_VALUE:
      llvm::outs() << "Global Value";
      break;
    case OperandType::NO_CFI_VALUE:
      llvm::outs() << "No CFI Value";
      break;
    case OperandType::NOT_SUPPORTED_OPERAND:
      llvm::outs() << "Not Supported Operand";
      break;
    case OperandType::VARIABLE:
      llvm::outs() << "Variable";
      break;
    case OperandType::NULL_OPERAND:
      llvm::outs() << "Null Operand";
      break;
    default:
      llvm::outs() << "Unknown";
      break;
    }

    llvm::outs() << "\n";
  }
  llvm::outs() << "Operands: ";

  llvm::outs() << "lhs: "; //<<instruction->getLHS();
  std::pair<SLIMOperand *, int> operandPair = instruction->getLHS();
  SLIMOperand *slimOperand = operandPair.first;
  // if (slimOperand) {
  //   GenSet instruction_genset;
  //   slimOperand->printOperand(llvm::outs());
  //   // addDefinition(slimOperand->getOnlyName().str());
  //   instruction_genset.definitions[slimOperand->getOnlyName().str()] =
  //       definitionSets[slimOperand->getOnlyName().str()].size();
  //   printGenSet(instruction_genset);

  //   KillSet instruction_killset;
  //   instruction_killset.definitions[slimOperand->getOnlyName().str()] =
  //       definitionSets[slimOperand->getOnlyName().str()];
  //   printKillSet(instruction_killset);
  //   // llvm::outs()<<slimOperand->getOnlyName();
  // } else {
  //   llvm::outs() << "Unknown Operand";
  // }
  llvm::outs() << "\n";
  if (instruction->getInstructionType() != InstructionType::ALLOCA) {

    for (unsigned i = 0; i < instruction->getNumOperands(); ++i) {
      if (i > 0) {
        llvm::outs() << ", ";
      }

      std::pair<SLIMOperand *, int> operandPair = instruction->getOperand(i);
      SLIMOperand *slimOperand = operandPair.first;

      if (slimOperand) {
        slimOperand->printOperand(llvm::outs());
      } else {
        llvm::outs() << "Unknown Operand";
      }
    }

  } else if (instruction->getInstructionType() == InstructionType::ALLOCA) {
    AllocaInstruction *allocaInst =
        dynamic_cast<AllocaInstruction *>(instruction);

    if (allocaInst) {
      // llvm::outs() << "Instruction Type: ALLOCA\n";

      // Access and print the operand of the AllocaInstruction
      std::pair<SLIMOperand *, int> resultOperand =
          allocaInst->getResultOperand();
      if (resultOperand.first != nullptr) {
        llvm::outs() << "Operand: ";
        resultOperand.first->printOperand(llvm::outs());
        llvm::outs() << "\n";
      }
    } else {
      llvm::outs() << "Failed to cast to AllocaInstruction\n";
    }
  }
  llvm::outs() << "\n";

  switch (instruction->getInstructionType()) {
  case InstructionType::BINARY_OPERATION:

    SLIMBinaryOperator binaryOperator =
        dynamic_cast<BinaryOperation *>(instruction)->getOperationType();

    // Map the binary operator type to its corresponding string representation
    std::map<SLIMBinaryOperator, std::string> operatorStrings = {
        {ADD, "+"},
        {SUB, "-"},
        {MUL, "*"},
        {DIV, "/"},
        {REM, "%"},
        {SHIFT_LEFT, "<<"},
        {LOGICAL_SHIFT_RIGHT, ">>>"},
        {ARITHMETIC_SHIFT_RIGHT, ">>"},
        {BITWISE_AND, "&"},
        {BITWISE_OR, "|"},
        {BITWISE_XOR, "^"}};

    // Print the binary operator
    llvm::outs() << "Binary Operation: " << operatorStrings[binaryOperator]
                 << "\n";

    // Now you can call the original printInstruction method without any changes
    // instruction->printInstruction();
  }
}

// Function to print instruction details
void slim::IR ::getVariableDetails(BaseInstruction *instruction) {

  std::pair<SLIMOperand *, int> operandPair = instruction->getLHS();
  SLIMOperand *slimOperand = operandPair.first;
  if (slimOperand) {
    slimOperand->printOperand(llvm::outs());
    addinitialDefinition(slimOperand->getOnlyName().str());
  } else {
    llvm::outs() << "Unknown Operand";
  }
  llvm::outs() << "\n";
}

void InOut(BaseInstruction *instruction, int index,
           BasicBlocksInOutInformation &b, GenSet &gen, KillSet &kill) {
  // llvm::outs() << "lhs: "; //<<instruction->getLHS();
  llvm::outs() << "\n---------\n";
  instruction->printInstruction();

  std::pair<SLIMOperand *, int> operandPair = instruction->getLHS();
  SLIMOperand *slimOperand = operandPair.first;
  if (slimOperand) {
    GenSet instruction_genset;
    addDefinition(slimOperand->getOnlyName().str());
    // slimOperand->printOperand(llvm::outs());
    instruction_genset.definitions[slimOperand->getOnlyName().str()] =
        definitionSets[slimOperand->getOnlyName().str()].size() - 1;
    Definition def1 = {
        slimOperand->getOnlyName().str(),
        (int)definitionSets[slimOperand->getOnlyName().str()].size()};

    // if(gen.definitions[slimOperand->getOnlyName().str()]){
    gen.definitions[slimOperand->getOnlyName().str()] =
        (int)definitionSets[slimOperand->getOnlyName().str()].size() - 1;
    // }

    // b.addToOutSet(index, def1);
    // llvm::outs()<<"here";
    printGenSet(instruction_genset);

    KillSet instruction_killset;
    instruction_killset.definitions[slimOperand->getOnlyName().str()] =
        definitionSets[slimOperand->getOnlyName().str()];
    kill.definitions[slimOperand->getOnlyName().str()] =
        instruction_killset.definitions[slimOperand->getOnlyName().str()];
    // for (vector_defn : definitionSets[slimOperand->getOnlyName().str()]) {
    //   kill.addToInSet(0,{slimOperand->getOnlyName().str(),vector_defn})
    // }

    printKillSet(instruction_killset);
    // llvm::outs()<<slimOperand->getOnlyName();
  } else {
    llvm::outs() << "Unknown Operand";
  }
}

void printBlockPredecessors() {
  for (const auto &entry : blockPredecessors) {
    int blockIndex = entry.first;
    const std::vector<int> &predecessors = entry.second;

    std::cout << "Predecessors of Basic Block " << blockIndex << ": ";
    for (int predecessor : predecessors) {
      std::cout << predecessor << " ";
    }
    std::cout << std::endl;
  }
}
void slim::IR::FillPredDetails(llvm::BasicBlock *basic_block,
                               int index_of_basic_block) {

  for (auto pred = llvm::pred_begin(basic_block);
       pred != llvm::pred_end(basic_block); pred++) {

    // Get the index of the predecessor basic block
    int predIndex = this->getBasicBlockId(*pred);

    // Update the map with predecessors for the current basic block
    blockPredecessors[index_of_basic_block].push_back(predIndex);

    if (std::next(pred) != llvm::pred_end(basic_block)) {
      llvm::outs() << ",";
    }
  }

  llvm::outs() << "])\n";
}

void Initialize_In_Out(BasicBlocksInOutInformation &b) {

  // Iterate through the definition sets and add variables with definition
  // number 0
  for (const auto &variableDefinitions : definitionSets) {
    const std::string &variableName = variableDefinitions.first;

    b.addToInSet(0, {variableName, 0});
  }
}

bool converged1(BasicBlocksInOutInformation b[], int num_iterations) {
  if (num_iterations < 2) {
    return false;
  }
  BasicBlocksInOutInformation b1 = b[num_iterations - 2];
  BasicBlocksInOutInformation b2 = b[num_iterations - 1];
  return b1 == b2;
}
#include <fstream> // Include the necessary header for file operations

void real_computing(GenSet g[], KillSet k[], int num_basic_blocks,
                    BasicBlocksInOutInformation inout[]) {

  int in_out_index = 1;

  BasicBlocksInOutInformation analysis(num_basic_blocks);
  bool converged = false;
  while (!converged) {
    converged = true;
    // analysis.InSets[0] = inout[in_out_index - 1].InSets[0];
    // analysis.OutSets[0] = inout[in_out_index - 1].OutSets[0];
    for (int i = 0; i < num_basic_blocks; i++) {
      /*
            In Set is Done Here
      */
      if (i != 0) {
        if (blockPredecessors.find(i) != blockPredecessors.end()) {
          // Key found, so extract and print the associated vector
          std::vector<int> &pred_vector = blockPredecessors[i];

          for (int pred_index : pred_vector) {
            for (const Definition &definition : analysis.OutSets[pred_index]) {
              analysis.addToInSet(i, definition);
            }
          }
          std::cout << std::endl;
        } else {
          // Key not found
          std::cout << "IndexKey " << i << " not found in the map."
                    << std::endl;
        }
      } else {
        analysis.InSets[i] = inout[in_out_index - 1].InSets[i];
      }

      if (inout[in_out_index - 1].InSets[i] != analysis.InSets[i]) {
        converged = false;
      }
      /*
            Out Set needs to be handled here
      */
      GenSet gen;
      gen = g[i];
      OurSet Inn;
      for (const Definition &definition : analysis.InSets[i]) {
        Inn.definitions[definition.variableName].push_back(definition.index);
      }
      llvm::outs() << "\n In of the Basic Block \n";
      printOurSet(Inn, "In of Basic Block");

      OurSet kill;
      kill.definitions = k[i].definitions;
      // Perform set difference: kill1 - kill2
      for (const auto &entry : kill.definitions) {
        const std::string &variable = entry.first;

        if (Inn.definitions.find(variable) != Inn.definitions.end()) {
          // The variable exists in both kill1 and kill2, perform set difference
          for (const int definition : entry.second) {
            auto &definitions1 = Inn.definitions[variable];
            auto it = std::remove(definitions1.begin(), definitions1.end(),
                                  definition);
            definitions1.erase(it, definitions1.end());
          }
        }
      }
      OurSet Inn_Killn = Inn;
      llvm::outs() << "\n Inn-Killn =\n";
      printOurSet(Inn, "Inn-Killn =");

      for (const auto &entry : gen.definitions) {
        const std::string &variable = entry.first;
        Inn_Killn.definitions[variable].push_back(entry.second);
      }

      llvm::outs() << "\n (Genn) U (Inn-Killn) =\n";

      printOurSet(Inn_Killn, "(Genn) U (Inn-Killn)");

      /*
      Filling OutSet
      */
      //llvm::outs() << "\n2-----" << i << "\n";
      for (const auto &entry : Inn_Killn.definitions) {
        const std::string &variable = entry.first;
        for (const int definition : entry.second) {
          analysis.addToOutSet(i, {variable, definition});
        }
      }

      /*
          std::vector<std::unordered_set<Definition>> InSets;
        std::vector<std::unordered_set<Definition>> OutSets;

          if(inout[in_out_index-1][i] != analysis[i]){
            converged = false;
          }
          */
      //llvm::outs() << "\n1-----" << i << "\n";
      analysis.printOutSet(i);
      //llvm::outs() << "\n4--------\n";
      inout[in_out_index - 1].printOutSet(i);
      //llvm::outs() << "\n5--------\n";
      if (inout[in_out_index - 1].OutSets[i] != analysis.OutSets[i]) {
        converged = false;
      }
      //llvm::outs() << "\n3-----" << in_out_index << "\n";
      // inout[in_out_index].InSets = analysis.InSets;
      // inout[in_out_index].OutSets = analysis.OutSets;

      //llvm::outs() << "\n-----" << i << "\n";
      analysis.printOutSet(i);
    }
    inout[in_out_index] = analysis;
    in_out_index++;
  }

  int num_iterations = in_out_index;
  for (int i = 0; i < num_basic_blocks; ++i) {
    inout[num_iterations - 2].printInSet(i);
    inout[num_iterations - 2].printOutSet(i);
  }

  for (int i = 0; i < num_basic_blocks; ++i) {
    inout[num_iterations - 1].printInSet(i);
    inout[num_iterations - 1].printOutSet(i);
  }
  // Create and open an HTML file for writing
  std::ofstream htmlFile("output.html");

  // Check if the file is open and write the HTML content
  if (htmlFile.is_open()) {
    // Write the HTML header
    htmlFile
        << "<html><head><title>InSet and OutSet Table</title></head><body>\n";

    // Create a table for GenSet and KillSet
    htmlFile << "<table border='1'>\n";
    htmlFile
        << "<tr><th>Block Number</th><th>GenSet</th><th>KillSet</th></tr>\n";

    for (int block = 0; block < num_basic_blocks; ++block) {
      htmlFile << "<tr><td>Block " << block << "</td>";

      // Write GenSet details
      htmlFile << "<td>";
      for (const auto &entry : g[block].definitions) {
        htmlFile << entry.first << "__";
        //for (const int index : entry.second) {
          int def = entry.second;
          htmlFile << def << ",";
        //}
        htmlFile << "<br>";
      }
      htmlFile << "</td>";

      // Write KillSet details
      htmlFile << "<td>";
      for (const auto &entry : k[block].definitions) {
        
        for (const int index : entry.second) {
          htmlFile << entry.first << "__";
          int def = index;
          htmlFile << def << ",";htmlFile << "<br>";
        }
        htmlFile << "<br>";
      }
      htmlFile << "</td>";

      htmlFile << "</tr>\n";
    }

    // Close the table for GenSet and KillSet
    htmlFile << "</table>\n";

    // Create a container div to hold all the tables side by side
    htmlFile << "<div style='display: flex;'>\n";

    for (int iteration = 0; iteration < in_out_index - 1; ++iteration) {
      // Create a div for each iteration
      htmlFile << "<div style='margin-right: 20px;'>\n";

      // Write a section header for each iteration
      htmlFile << "<h2>Iteration " << iteration << "</h2>\n";

      // Create a table to hold the data for this iteration
      htmlFile << "<table border='1'>\n";

      // Write the table header for this iteration
      htmlFile << "<tr><th>Block</th><th>InSet</th><th>OutSet</th></tr>\n";

      for (int block = 0; block < num_basic_blocks; ++block) {
        // Write InSet and OutSet details for each block
        htmlFile << "<tr><td>Block " << block << "</td>";

        htmlFile << "<td>";
        std::unordered_set<Definition> &inSet = inout[iteration].InSets[block];
        for (const Definition &def : inSet) {
          htmlFile << def.variableName << "__" << def.index << "<br>";
        }
        htmlFile << "</td>";

        htmlFile << "<td>";
        std::unordered_set<Definition> &outSet =
            inout[iteration].OutSets[block];
        for (const Definition &def : outSet) {
          htmlFile << def.variableName << "__" << def.index << "<br>";
        }
        htmlFile << "</td>";

        htmlFile << "</tr>\n";
      }

      // Close the table for this iteration
      htmlFile << "</table>\n";

      // Close the div for this iteration
      htmlFile << "</div>\n";
    }

    // Close the container div
    htmlFile << "</div>\n";

    // Close the HTML body and document
    htmlFile << "</body></html>\n";

    // Close the HTML file
    htmlFile.close();
  } else {
    std::cerr << "Failed to open the HTML file for writing." << std::endl;
  }
}
void slim::IR::dumpIR() {
  std::unordered_map<llvm::Function *, bool> func_visited;

  int numBasicBlocks = 0;
  /*
  Getting predecessors detail
  */
  for (auto &entry : this->func_bb_to_inst_id) {
    llvm::Function *func = entry.first.first;
    llvm::BasicBlock *basic_block = entry.first.second;
    FillPredDetails(basic_block, numBasicBlocks);
    numBasicBlocks++;
  }

  // Create an instance of BasicBlocksInOutInformation
  BasicBlocksInOutInformation analysis(numBasicBlocks);
  GenSet global_genset[numBasicBlocks];
  KillSet global_killset[numBasicBlocks];
  int num_iterations = 0;
  // BasicBlocksInOutInformation in_out_per_iteration[100](numBasicBlocks);
  BasicBlocksInOutInformation in_out_per_iteration[15]; // Declare an array

  for (int i = 0; i < 15; ++i) {
    in_out_per_iteration[i] = BasicBlocksInOutInformation(numBasicBlocks);
  }
  llvm::outs() << "\n Number of Basic Blocks =" << numBasicBlocks << "\n";
  for (auto &entry : this->func_bb_to_inst_id) {
    llvm::Function *func = entry.first.first;
    llvm::BasicBlock *basic_block = entry.first.second;
    for (long long instruction_id : entry.second) {
      BaseInstruction *instruction = inst_id_to_object[instruction_id];
      getVariableDetails(instruction); // getting all the variable definitions
    }
  }
  Initialize_In_Out(analysis); // In0 as BI all others = {}
  in_out_per_iteration[num_iterations] = analysis;
  num_iterations = num_iterations + 1;

  // definitionSets.clear();
  // traversing each basic block single time
  for (auto &entry : this->func_bb_to_inst_id) {
    llvm::Function *func = entry.first.first;
    llvm::BasicBlock *basic_block = entry.first.second;

    printFunctionDetails(func, func_visited);

    printBasicBlockDetails(basic_block, func_visited);

    for (long long instruction_id : entry.second) {
      BaseInstruction *instruction = inst_id_to_object[instruction_id];
      printInstructionDetails(instruction);
    }
  }
  int index = 0;
  for (auto &entry : this->func_bb_to_inst_id) {
    // for getting gen and kill sets
    for (long long instruction_id : entry.second) {
      BaseInstruction *instruction = inst_id_to_object[instruction_id];
      InOut(instruction, index, analysis, global_genset[index],
            global_killset[index]);
    }
    index++;
  }
  // llvm::outs()<<"We can reach here\n";
  real_computing(global_genset, global_killset, numBasicBlocks,
                 in_out_per_iteration);
  // in_out_per_iteration[num_iterations] = analysis;
  // num_iterations = num_iterations + 1;
  llvm::outs() << "\nDefinitionSets\n";
  printDefinitionSets();
  llvm::outs() << "\nBlock Predecessors\n";
  printBlockPredecessors();
  // Print InSet and OutSet for each block
  // for (int i = 0; i < numBasicBlocks; ++i) {
  //   analysis.printInSet(i);
  //   analysis.printOutSet(i);
  // }

  // llvm::outs() << "Last 2 In and Out2"
  //              << "\n";
  // for (int i = 0; i < numBasicBlocks; ++i) {
  //   in_out_per_iteration[num_iterations - 2].printInSet(i);
  //   in_out_per_iteration[num_iterations - 2].printOutSet(i);
  // }

  // for (int i = 0; i < numBasicBlocks; ++i) {
  //   in_out_per_iteration[num_iterations - 1].printInSet(i);
  //   in_out_per_iteration[num_iterations - 1].printOutSet(i);
  // }
}

// void slim::IR::dumpIR() {
//   std::unordered_map<llvm::Function *, bool> func_visited;

//   int numBasicBlocks = 0;
//   for (auto &entry : this->func_bb_to_inst_id) {
//     llvm::Function *func = entry.first.first;
//     llvm::BasicBlock *basic_block = entry.first.second;
//     FillPredDetails(basic_block, numBasicBlocks);
//     numBasicBlocks++;
//   }

//   // Create an instance of BasicBlocksInOutInformation
//   BasicBlocksInOutInformation analysis(numBasicBlocks);
//   int num_iterations = 0;
//   BasicBlocksInOutInformation in_out_per_iteration[100](numBasicBlocks);

//   llvm::outs() << "\n Number of Basic Blocks =" << numBasicBlocks << "\n";
//   for (auto &entry : this->func_bb_to_inst_id) {
//     llvm::Function *func = entry.first.first;
//     llvm::BasicBlock *basic_block = entry.first.second;
//     for (long long instruction_id : entry.second) {
//       BaseInstruction *instruction = inst_id_to_object[instruction_id];
//       getVariableDetails(instruction);
//     }
//   }
//   while (!converged(in_out_per_iteration[num_iterations], analysis)) {
//     int index = 0;
//     definitionSets.clear();
//     for (auto &entry : this->func_bb_to_inst_id) {
//       llvm::Function *func = entry.first.first;
//       llvm::BasicBlock *basic_block = entry.first.second;

//       printFunctionDetails(func, func_visited);

//       printBasicBlockDetails(basic_block, func_visited);

//       /*Before each Basic Block*/
//       /*
//             In Set is Done Here
//       */
//       if (index == 0) {
//         Initialize_In_Out(analysis);
//       } else {
//         for (auto pred = llvm::pred_begin(basic_block);
//              pred != llvm::pred_end(basic_block); pred++) {

//           // Get the index of the predecessor basic block
//           int predIndex = this->getBasicBlockId(*pred);

//           // Update the map with predecessors for the current basic block
//           // blockPredecessors[index_of_basic_block].push_back(predIndex);
//           for (const Definition &definition : analysis.OutSets[predIndex]) {
//             analysis.addToInSet(index, definition);
//           }
//         }
//       }
//       /*
//             Out Set needs to be handled here
//       */
//       GenSet local_gen;
//       KillSet local_kill;

//       for (long long instruction_id : entry.second) {
//         BaseInstruction *instruction = inst_id_to_object[instruction_id];
//         printInstructionDetails(instruction);
//       }

//       for (long long instruction_id : entry.second) {
//         BaseInstruction *instruction = inst_id_to_object[instruction_id];
//         InOut(instruction, index, analysis, local_gen, local_kill);
//       }
//       /*
//       Now we have GenSet and KillSet for each block
//       */
//       llvm::outs() << "\n\n--------------Local Gen Set--------------\n\n";
//       printGenSet(local_gen);
//       llvm::outs() << "\n\n--------------Local Kill Set--------------\n\n";
//       printKillSet(local_kill);
//       llvm::outs() << "\n";

//       GenSet temp;
//       temp = local_gen;
//       KillSet temp2;
//       for (const Definition &definition : analysis.InSets[index]) {
//         temp2.definitions[definition.variableName].push_back(definition.index);
//         // std::cout << definition.variableName << "_" << definition.index <<
//         // ",";
//       }
//       llvm::outs() << "\n In of the Basic Block \n";
//       printKillSet(temp2);

//       // Define two KillSet instances
//       KillSet kill1 = temp2;
//       KillSet kill2 = local_kill;

//       // Fill the kill1 and kill2 structures with data

//       // Perform set difference: kill1 - kill2
//       for (const auto &entry : kill2.definitions) {
//         const std::string &variable = entry.first;

//         if (kill1.definitions.find(variable) != kill1.definitions.end()) {
//           // The variable exists in both kill1 and kill2, perform set
//           difference for (const int definition : entry.second) {
//             auto &definitions1 = kill1.definitions[variable];
//             auto it = std::remove(definitions1.begin(), definitions1.end(),
//                                   definition);
//             definitions1.erase(it, definitions1.end());
//           }
//         }
//       }
//       llvm::outs() << "\n Inn-Killn =\n";
//       printKillSet(kill1);

//       for (const auto &entry : local_gen.definitions) {
//         const std::string &variable = entry.first;
//         kill1.definitions[variable].push_back(entry.second);
//       }

//       llvm::outs() << "\n (Genn) U (Inn-Killn) =\n";
//       printKillSet(kill1);

//       for (const auto &entry : kill1.definitions) {
//         const std::string &variable = entry.first;
//         for (const int definition : entry.second) {
//           analysis.addToOutSet(index, {variable, definition});
//         }
//       }
//       analysis.printOutSet(index);

//       printDefinitionSets();
//       index++;
//     }
//     in_out_per_iteration[num_iterations]
//   }

//   printBlockPredecessors();
//   // Print InSet and OutSet for each block
//   for (int i = 0; i < numBasicBlocks; ++i) {
//     analysis.printInSet(i);
//     analysis.printOutSet(i);
//   }
// }

// void slim::IR::dumpIR() {
//   // Keeps track whether the function details have been printed or not
//   std::unordered_map<llvm::Function *, bool> func_visited;

//   // Iterate over the function basic block map
//   for (auto &entry : this->func_bb_to_inst_id) {
//     llvm::Function *func = entry.first.first;
//     llvm::BasicBlock *basic_block = entry.first.second;

//     // Check if the function details are already printed and if not, print
//     the
//     // details and mark the function as visited
//     if (func_visited.find(func) == func_visited.end()) {
//       if (func->getSubprogram())
//         llvm::outs() << "[" << func->getSubprogram()->getFilename() << "] ";

//       llvm::outs() << "Function: " << func->getName() << "\n";
//       llvm::outs() << "A-------------------------------------A"
//                    << "\n";

//       // Mark the function as visited
//       func_visited[func] = true;
//     }

//     // Print the basic block name
//     llvm::outs() << "Basic block " << this->getBasicBlockId(basic_block) <<
//     ": "
//                  << basic_block->getName() << " (Predecessors::: ";
//     llvm::outs() << "[";

//     // Print the names of predecessor basic blocks
//     for (auto pred = llvm::pred_begin(basic_block);
//          pred != llvm::pred_end(basic_block); pred++) {
//       llvm::outs() << (*pred)->getName();

//       if (std::next(pred) != ((llvm::pred_end(basic_block)))) {
//         llvm::outs() << ",,, ";
//       }
//     }

//     llvm::outs() << "])\n";

//     for (long long instruction_id : entry.second) {

//       llvm::outs() <<
//       "----------------------------------------------------------"<<"\n";

//       BaseInstruction *instruction = inst_id_to_object[instruction_id];
//       // Print the instruction type
//       //llvm::outs() << "Instruction Type: " //<<
//       instruction->getInstructionType()
//                   // << "\n";
//       instruction->printInstruction();
//       llvm::outs() << "Instruction Type: " ;
//       switch (instruction->getInstructionType()) {

//       case InstructionType::ALLOCA:
//         llvm::outs() << "ALLOCA\n";
//         break;
//       case InstructionType::LOAD:
//         llvm::outs() << "LOAD\n";
//         break;
//       case InstructionType::STORE:
//         llvm::outs() << "STORE\n";
//         break;
//       case InstructionType::FENCE:
//         llvm::outs() << "FENCE\n";
//         break;
//       case InstructionType::ATOMIC_COMPARE_CHANGE:
//         llvm::outs() << "ATOMIC_COMPARE_CHANGE\n";
//         break;
//       case InstructionType::ATOMIC_MODIFY_MEM:
//         llvm::outs() << "ATOMIC_MODIFY_MEM\n";
//         break;
//       case InstructionType::GET_ELEMENT_PTR:
//         llvm::outs() << "GET_ELEMENT_PTR\n";
//         break;
//       case InstructionType::FP_NEGATION:
//         llvm::outs() << "FP_NEGATION\n";
//         break;
//       case InstructionType::BINARY_OPERATION: {
//         llvm::outs() << "BINARY_OPERATION\n";
//         // Handle binary operation here
//         /*if (auto binaryOp = dynamic_cast<BinaryOperation *>(instruction)) {
//           // Call the printInstruction method for binary operation
//           binaryOp->printInstruction();
//         }*/
//       } break;
//       case InstructionType::EXTRACT_ELEMENT:
//         llvm::outs() << "EXTRACT_ELEMENT\n";
//         break;
//       case InstructionType::INSERT_ELEMENT:
//         llvm::outs() << "INSERT_ELEMENT\n";
//         break;
//       case InstructionType::SHUFFLE_VECTOR:
//         llvm::outs() << "SHUFFLE_VECTOR\n";
//         break;
//       case InstructionType::EXTRACT_VALUE:
//         llvm::outs() << "EXTRACT_VALUE\n";
//         break;
//       case InstructionType::INSERT_VALUE:
//         llvm::outs() << "INSERT_VALUE\n";
//         break;
//       case InstructionType::TRUNC:
//         llvm::outs() << "TRUNC\n";
//         break;
//       case InstructionType::ZEXT:
//         llvm::outs() << "ZEXT\n";
//         break;
//       case InstructionType::SEXT:
//         llvm::outs() << "SEXT\n";
//         break;
//       case InstructionType::FPEXT:
//         llvm::outs() << "FPEXT\n";
//         break;
//       case InstructionType::FP_TO_INT:
//         llvm::outs() << "FP_TO_INT\n";
//         break;
//       case InstructionType::INT_TO_FP:
//         llvm::outs() << "INT_TO_FP\n";
//         break;
//       case InstructionType::PTR_TO_INT:
//         llvm::outs() << "PTR_TO_INT\n";
//         break;
//       case InstructionType::INT_TO_PTR:
//         llvm::outs() << "INT_TO_PTR\n";
//         break;
//       case InstructionType::BITCAST:
//         llvm::outs() << "BITCAST\n";
//         break;
//       case InstructionType::ADDR_SPACE:
//         llvm::outs() << "ADDR_SPACE\n";
//         break;
//       case InstructionType::COMPARE:
//         llvm::outs() << "COMPARE\n";
//         break;
//       case InstructionType::PHI:
//         llvm::outs() << "PHI\n";
//         break;
//       case InstructionType::SELECT:
//         llvm::outs() << "SELECT\n";
//         break;
//       case InstructionType::FREEZE:
//         llvm::outs() << "FREEZE\n";
//         break;
//       case InstructionType::CALL:
//         llvm::outs() << "CALL\n";
//         break;
//       case InstructionType::VAR_ARG:
//         llvm::outs() << "VAR_ARG\n";
//         break;
//       case InstructionType::LANDING_PAD:
//         llvm::outs() << "LANDING_PAD\n";
//         break;
//       case InstructionType::CATCH_PAD:
//         llvm::outs() << "CATCH_PAD\n";
//         break;
//       case InstructionType::CLEANUP_PAD:
//         llvm::outs() << "CLEANUP_PAD\n";
//         break;
//       case InstructionType::RETURN:
//         llvm::outs() << "RETURN\n";
//         break;
//       case InstructionType::BRANCH:
//         llvm::outs() << "BRANCH\n";
//         break;
//       case InstructionType::SWITCH:
//         llvm::outs() << "SWITCH\n";
//         break;
//       case InstructionType::INDIRECT_BRANCH:
//         llvm::outs() << "INDIRECT_BRANCH\n";
//         break;
//       case InstructionType::INVOKE:
//         llvm::outs() << "INVOKE\n";
//         break;
//       case InstructionType::CALL_BR:
//         llvm::outs() << "CALL_BR\n";
//         break;
//       case InstructionType::RESUME:
//         llvm::outs() << "RESUME\n";
//         break;
//       case InstructionType::CATCH_SWITCH:
//         llvm::outs() << "CATCH_SWITCH\n";
//         break;
//       case InstructionType::CATCH_RETURN:
//         llvm::outs() << "CATCH_RETURN\n";
//         break;
//       case InstructionType::CLEANUP_RETURN:
//         llvm::outs() << "CLEANUP_RETURN\n";
//         break;
//       case InstructionType::UNREACHABLE:
//         llvm::outs() << "UNREACHABLE\n";
//         break;
//       case InstructionType::OTHER:
//         llvm::outs() << "OTHER\n";
//         break;
//       case InstructionType::NOT_ASSIGNED:
//         llvm::outs() << "NOT_ASSIGNED\n";
//         break;
//       }
//       //instruction->printInstruction();
//       if (instruction->getInstructionType() < 33) {
//         // Determine the instruction type and print it
//         OperandType operandType =
//             instruction->getResultOperand().first->getOperandType();
//         switch (operandType) {
//         case OperandType::GEP_OPERATOR:
//           llvm::outs() << "GEP Operator";
//           break;
//         case OperandType::ADDR_SPACE_CAST_OPERATOR:
//           llvm::outs() << "AddrSpaceCast Operator";
//           break;
//         case OperandType::BITCAST_OPERATOR:
//           llvm::outs() << "Bitcast Operator";
//           break;
//         case OperandType::PTR_TO_INT_OPERATOR:
//           llvm::outs() << "PtrToInt Operator";
//           break;
//         case OperandType::ZEXT_OPERATOR:
//           llvm::outs() << "ZExt Operator";
//           break;
//         case OperandType::FP_MATH_OPERATOR:
//           llvm::outs() << "FP Math Operator";
//           break;
//         case OperandType::BLOCK_ADDRESS:
//           llvm::outs() << "Block Address";
//           break;
//         case OperandType::CONSTANT_AGGREGATE:
//           llvm::outs() << "Constant Aggregate";
//           break;
//         case OperandType::CONSTANT_DATA_SEQUENTIAL:
//           llvm::outs() << "Constant Data Sequential";
//           break;
//         case OperandType::CONSTANT_POINTER_NULL:
//           llvm::outs() << "Constant Pointer Null";
//           break;
//         case OperandType::CONSTANT_TOKEN_NONE:
//           llvm::outs() << "Constant Token None";
//           break;
//         case OperandType::UNDEF_VALUE:
//           llvm::outs() << "Undef Value";
//           break;
//         case OperandType::CONSTANT_INT:
//           llvm::outs() << "Constant Int";
//           break;
//         case OperandType::CONSTANT_FP:
//           llvm::outs() << "Constant FP";
//           break;
//         case OperandType::DSO_LOCAL_EQUIVALENT:
//           llvm::outs() << "DSO Local Equivalent";
//           break;
//         case OperandType::GLOBAL_VALUE:
//           llvm::outs() << "Global Value";
//           break;
//         case OperandType::NO_CFI_VALUE:
//           llvm::outs() << "No CFI Value";
//           break;
//         case OperandType::NOT_SUPPORTED_OPERAND:
//           llvm::outs() << "Not Supported Operand";
//           break;
//         case OperandType::VARIABLE:
//           llvm::outs() << "Variable";
//           break;
//         case OperandType::NULL_OPERAND:
//           llvm::outs() << "Null Operand";
//           break;
//         default:
//           llvm::outs() << "Unknown";
//           break;
//         }

//         llvm::outs() << "\n";
//       }
//       llvm::outs() << "Operands: ";
//       for (unsigned i = 0; i < instruction->getNumOperands(); ++i) {
//         if (i > 0) {
//           llvm::outs() << ", ";
//         }

//         std::pair<SLIMOperand *, int> operandPair =
//         instruction->getOperand(i); SLIMOperand *slimOperand =
//         operandPair.first;

//         if (slimOperand) {
//           slimOperand->printOperand(llvm::outs());
//         } else {
//           llvm::outs() << "Unknown Operand";
//         }
//       }
//       llvm::outs() << "\n";
//       switch (instruction->getInstructionType()) {
//       case InstructionType::BINARY_OPERATION:

//         SLIMBinaryOperator binaryOperator =
//             dynamic_cast<BinaryOperation *>(instruction)->getOperationType();

//         // Map the binary operator type to its corresponding string
//         // representation
//         std::map<SLIMBinaryOperator, std::string> operatorStrings = {
//             {ADD, "+"},
//             {SUB, "-"},
//             {MUL, "*"},
//             {DIV, "/"},
//             {REM, "%"},
//             {SHIFT_LEFT, "<<"},
//             {LOGICAL_SHIFT_RIGHT, ">>>"},
//             {ARITHMETIC_SHIFT_RIGHT, ">>"},
//             {BITWISE_AND, "&"},
//             {BITWISE_OR, "|"},
//             {BITWISE_XOR, "^"}};

//         // Print the binary operator
//         llvm::outs() << "Binary Operation: " <<
//         operatorStrings[binaryOperator]
//                      << "\n";

//         // Now you can call the original printInstruction method without any
//         // changes
//         // instruction->printInstruction();
//       }
//     }
//     llvm::outs() << "\n";
//   }
//   llvm::outs() << "\n\n";
// }*/
void slim::IR::generateIR() {

  // Fetch the function list of the module
  llvm::SymbolTableList<llvm::Function> &function_list =
      this->llvm_module->getFunctionList();

  // For each function in the module
  for (llvm::Function &function : function_list) {
    // Append the pointer to the function to the "functions" list
    if (!function.isIntrinsic() && !function.isDeclaration()) {
      this->functions.push_back(&function);
    } else {
      continue;
    }

    // For each basic block in the function
    for (llvm::BasicBlock &basic_block : function.getBasicBlockList()) {
      // Create function-basicblock pair
      std::pair<llvm::Function *, llvm::BasicBlock *> func_basic_block{
          &function, &basic_block};

      this->basic_block_to_id[&basic_block] = this->total_basic_blocks;

      this->total_basic_blocks++;

      // For each instruction in the basic block
      for (llvm::Instruction &instruction : basic_block.getInstList()) {
        if (instruction.hasMetadataOtherThanDebugLoc() ||
            instruction.isDebugOrPseudoInst()) {
          continue;
        }

        BaseInstruction *base_instruction =
            slim::processLLVMInstruction(instruction);

        if (base_instruction->getInstructionType() == InstructionType::CALL) {
          CallInstruction *call_instruction =
              (CallInstruction *)base_instruction;

          if (!call_instruction->isIndirectCall() &&
              !call_instruction->getCalleeFunction()->isDeclaration()) {
            for (unsigned arg_i = 0;
                 arg_i < call_instruction->getNumFormalArguments(); arg_i++) {
              llvm::Argument *formal_argument =
                  call_instruction->getFormalArgument(arg_i);
              SLIMOperand *formal_slim_argument =
                  OperandRepository::getSLIMOperand(formal_argument);

              if (!formal_slim_argument) {
                formal_slim_argument = new SLIMOperand(formal_argument);
                OperandRepository::setSLIMOperand(formal_argument,
                                                  formal_slim_argument);
              }

              LoadInstruction *new_load_instr = new LoadInstruction(
                  &llvm::cast<llvm::CallInst>(instruction),
                  formal_slim_argument,
                  call_instruction->getOperand(arg_i).first);

              // The initial value of total instructions is 0 and it is
              // incremented after every instruction
              long long instruction_id = this->total_instructions;

              // Increment the total instructions count
              this->total_instructions++;

              new_load_instr->setInstructionId(instruction_id);

              this->func_bb_to_inst_id[func_basic_block].push_back(
                  instruction_id);

              // Map the instruction id to the corresponding SLIM instruction
              this->inst_id_to_object[instruction_id] = new_load_instr;
            }
          }
        }

        // The initial value of total instructions is 0 and it is incremented
        // after every instruction
        long long instruction_id = this->total_instructions;

        // Increment the total instructions count
        this->total_instructions++;

        base_instruction->setInstructionId(instruction_id);

        this->func_bb_to_inst_id[func_basic_block].push_back(instruction_id);

        // Map the instruction id to the corresponding SLIM instruction
        this->inst_id_to_object[instruction_id] = base_instruction;
      }
    }
  }
}

// Returns the LLVM module
std::unique_ptr<llvm::Module> &slim::IR::getLLVMModule() {
  return this->llvm_module;
}

// Provides APIs similar to the older implementation of SLIM in order to support
// the implementations that are built using the older SLIM as a base
slim::LegacyIR::LegacyIR() { slim_ir = new slim::IR(); }

// Add the instructions of a basic block (of a given function) in the IR
void slim::LegacyIR::simplifyIR(llvm::Function *function,
                                llvm::BasicBlock *basic_block) {
  this->slim_ir->addFuncBasicBlockInstructions(function, basic_block);
}

// Get the repository (in the form of function-basicblock to instructions
// mappings) of all the SLIM instructions
std::map<std::pair<llvm::Function *, llvm::BasicBlock *>,
         std::list<long long>> &
slim::LegacyIR::getfuncBBInsMap() {
  return this->slim_ir->getFuncBBToInstructions();
}

// Get the instruction id to SLIM instruction map
std::map<long long, BaseInstruction *> &
slim::LegacyIR::getGlobalInstrIndexList() {
  return this->slim_ir->getIdToInstructionsMap();
}

// Returns the corresponding LLVM instruction for the instruction id
llvm::Instruction *slim::LegacyIR::getInstforIndx(long long index) {
  BaseInstruction *slim_instruction = this->slim_ir->getInstrFromIndex(index);

  return slim_instruction->getLLVMInstruction();
}
